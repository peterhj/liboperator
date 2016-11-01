use prelude::*;
use opt::{
  ClassOptStats,
};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::fs::{File};
use std::io::{Write};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone, Debug)]
pub struct AdagradConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub epsilon:      f32,
  //pub checkpoint:   Option<CheckpointConfig>,
}

pub struct AdagradWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  cfg:          AdagradConfig,
  //checkpoint:   CheckpointState,
  iter_counter: usize,
  operator:     Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  grad_sz:      usize,
  param:        Vec<f32>,
  param_saved:  Vec<f32>,
  grad:         Vec<f32>,
  grad_var_acc: Vec<f32>,
  ada_update:   Vec<f32>,
  tmp_buf:      Vec<f32>,
  stats_it:     usize,
  stats:        ClassOptStats,
  //_marker:      PhantomData<R>,
}

impl<S, Loss> AdagradWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  pub fn new(cfg: AdagradConfig, operator: Rc<RefCell<Loss>>) -> AdagradWorker<S, Loss> {
    let grad_sz = operator.borrow_mut().diff_param_sz();
    let cache = Vec::with_capacity(cfg.minibatch_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut ada_update = Vec::with_capacity(grad_sz);
    ada_update.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    /*let mut checkpoint = CheckpointState::default();
    if let Some(ref chk_cfg) = cfg.checkpoint {
      checkpoint = chk_cfg.build_state();
    }
    if let Some(ref mut config_file) = checkpoint.config_file {
      writeln!(config_file, "{:?}", cfg).unwrap();
    }*/
    AdagradWorker{
      cfg:          cfg,
      //checkpoint:   checkpoint,
      iter_counter: 0,
      operator:     operator,
      cache:        cache,
      grad_sz:      grad_sz,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      grad_var_acc: grad_var_acc,
      ada_update:   ada_update,
      tmp_buf:      tmp_buf,
      stats_it:     0,
      stats:        Default::default(),
      //_marker:      PhantomData,
    }
  }
}

impl<S, Loss> OptWorker<f32, S> for AdagradWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Rng = Xorshiftplus128Rng;

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.operator.borrow_mut().init_param(rng);
    self.operator.borrow_mut().store_diff_param(&mut self.param_saved);
  }

  /*fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
    unimplemented!();
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
    unimplemented!();
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
    unimplemented!();
  }*/

  fn step(&mut self, samples: &mut Iterator<Item=S>) {
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;

    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = self.iter_counter / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      _ => unimplemented!(),
    };

    self.cache.clear();
    for mut sample in samples.take(self.cfg.minibatch_sz) {
      self.cache.push(sample);
    }

    let mut operator = self.operator.borrow_mut();

    //operator.load_diff_param(&mut self.param_saved);
    operator.save_rng_state();
    operator.reset_loss();
    operator.reset_grad();
    operator.next_iteration();
    for batch in 0 .. num_batches {
      let batch_start = batch * self.cfg.batch_sz;
      let batch_end = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz);
      operator.load_batch(&self.cache[batch_start .. batch_end]);
      operator.forward(OpPhase::Learning);
      operator.backward();
    }
    operator.update_nondiff_param(self.iter_counter);

    operator.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    let loss = operator.store_loss() / self.cfg.minibatch_sz as f32;

    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).add(1.0, self.tmp_buf.reshape(self.grad_sz));

    self.ada_update.copy_from_slice(&self.grad_var_acc);
    assert!(self.cfg.epsilon * self.cfg.epsilon > 0.0);
    self.ada_update.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon * self.cfg.epsilon);
    self.ada_update.reshape_mut(self.grad_sz).sqrt();
    self.ada_update.reshape_mut(self.grad_sz).reciprocal();
    self.ada_update.reshape_mut(self.grad_sz).elem_mult(self.grad.reshape(self.grad_sz));

    self.param.copy_from_slice(&self.param_saved);
    self.param.reshape_mut(self.grad_sz).add(-step_size, self.ada_update.reshape(self.grad_sz));
    self.param_saved.copy_from_slice(&self.param);
    operator.load_diff_param(&mut self.param_saved);

    self.iter_counter += 1;

    /*if let Some(ref mut train_file) = self.checkpoint.train_file {
      // TODO
    }*/

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.correct_count += operator._store_accuracy();
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (loss - self.stats.avg_loss);
  }

  fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    let mut operator = self.operator.borrow_mut();
    self.cache.clear();
    //operator.load_diff_param(&mut self.param_saved);
    operator.reset_loss();
    for mut sample in samples.take(epoch_sz) {
      //sample.mix_weight(1.0 / epoch_sz as f32);
      self.cache.push(sample);
      if self.cache.len() == self.cfg.batch_sz {
        operator.load_batch(&self.cache);
        operator.forward(OpPhase::Inference);
        self.cache.clear();
      }
    }
    if self.cache.len() > 0 {
      operator.load_batch(&self.cache);
      operator.forward(OpPhase::Inference);
    }
    let loss = operator.store_loss() / epoch_sz as f32;
    self.stats_it += 1;
    self.stats.sample_count += epoch_sz;
    self.stats.correct_count += operator._store_accuracy();
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (loss - self.stats.avg_loss);
  }
}

impl<S, Loss> AdagradWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> + LossReport<ClassLossStats> {
  pub fn update_stats(&self, stats: &mut ClassLossStats) {
    let mut operator = self.operator.borrow_mut();
    operator.update_stats(self.iter_counter, stats);
  }
}

impl<S, Loss> OptStats<ClassOptStats> for AdagradWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  fn reset_opt_stats(&mut self) {
    self.stats_it = 0;
    self.stats.sample_count = 0;
    self.stats.correct_count = 0;
    self.stats.avg_loss = 0.0;
  }

  fn get_opt_stats(&self) -> &ClassOptStats {
    &self.stats
  }
}
