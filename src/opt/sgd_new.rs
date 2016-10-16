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
pub struct SgdConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub momentum:     Option<GradientMomentum>,
  pub checkpoint:   Option<CheckpointConfig>,
}

pub struct SgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  cfg:          SgdConfig,
  config_file:  Option<File>,
  trace_file:   Option<File>,
  iter_counter: usize,
  operator:     Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  grad_sz:      usize,
  param:        Vec<f32>,
  param_saved:  Vec<f32>,
  grad:         Vec<f32>,
  grad_acc:     Vec<f32>,
  stats_it:     usize,
  stats:        ClassOptStats,
  //_marker:      PhantomData<R>,
}

impl<S, Loss> SgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  pub fn new(cfg: SgdConfig, operator: Rc<RefCell<Loss>>) -> SgdWorker<S, Loss> {
    let grad_sz = operator.borrow_mut().diff_param_sz();
    let cache = Vec::with_capacity(cfg.minibatch_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    let (mut config_file, trace_file) = if let Some(ref checkpoint) = cfg.checkpoint {
      checkpoint.maybe_create_trace()
    } else {
      (None, None)
    };
    if let Some(ref mut config_file) = config_file {
      writeln!(config_file, "{:?}", cfg).unwrap();
    }
    SgdWorker{
      cfg:          cfg,
      config_file:  config_file,
      trace_file:   trace_file,
      iter_counter: 0,
      operator:     operator,
      cache:        cache,
      grad_sz:      grad_sz,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      //_marker:      PhantomData,
    }
  }
}

impl<S, Loss> OptWorker<f32, S> for SgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Rng = Xorshiftplus128Rng;

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.operator.borrow_mut().init_param(rng);
    self.operator.borrow_mut().store_diff_param(&mut self.param_saved);
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
    unimplemented!();
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
    unimplemented!();
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
    unimplemented!();
  }

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
      //sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
      self.cache.push(sample);
    }

    let mut operator = self.operator.borrow_mut();

    operator.save_rng_state();
    operator.reset_loss();
    operator.reset_grad();
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      //operator.update_diff_param(mu, 1.0, &mut self.grad_acc);
      operator.store_diff_param(&mut self.param);
      self.param.reshape_mut(self.grad_sz).add(mu, self.grad_acc.reshape(self.grad_sz));
      operator.load_diff_param(&mut self.param);
    }
    operator.next_iteration();
    for batch in 0 .. num_batches {
      let batch_start = batch * self.cfg.batch_sz;
      let batch_end = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz);
      operator.load_batch(&self.cache[batch_start .. batch_end]);
      operator.forward(OpPhase::Learning);
      operator.backward();
    }
    if let Some(GradientMomentum::Nesterov(_)) = self.cfg.momentum {
      operator.load_diff_param(&mut self.param_saved);
    }

    operator.update_nondiff_param(self.iter_counter);

    operator.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / self.cfg.minibatch_sz as f32);
    let loss = operator.store_loss() / self.cfg.minibatch_sz as f32;

    if let Some(GradientMomentum::HeavyBall(mu)) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.grad_sz).scale(mu);
    } else if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.grad_sz).scale(mu);
    } else {
      self.grad_acc.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.grad_acc.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));

    //operator.update_diff_param(1.0, 1.0, &mut self.grad_acc);
    operator.store_diff_param(&mut self.param);
    self.param.reshape_mut(self.grad_sz).add(1.0, self.grad_acc.reshape(self.grad_sz));
    operator.load_diff_param(&mut self.param);
    operator.store_diff_param(&mut self.param_saved);

    self.iter_counter += 1;

    if let Some(ref mut trace_file) = self.trace_file {
      // TODO
    }

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.correct_count += operator._store_accuracy();
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (loss - self.stats.avg_loss);
  }

  fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    let mut operator = self.operator.borrow_mut();
    self.cache.clear();
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

impl<S, Loss> OptStats<ClassOptStats> for SgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
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
