use prelude::*;
use opt::{
  ClassOptStats,
};
use opt::sgd_new::{SgdConfig};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};
use rng::xorshift::{Xorshiftplus128Rng};
use sharedmem::{SharedMem};
use sharedmem::sync::{SpinBarrier};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::fs::{File};
use std::io::{Write};
use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc, Mutex};

/*#[derive(Clone, Debug)]
pub struct SgdConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub momentum:     Option<GradientMomentum>,
  pub checkpoint:   Option<CheckpointConfig>,
}*/

#[derive(Clone)]
pub struct SharedSgdBuilder {
  cfg:          SgdConfig,
  num_workers:  usize,
  shared_bar:   Arc<SpinBarrier>,
  shared_grad:  Arc<Mutex<Vec<f32>>>,
}

impl SharedSgdBuilder {
  pub fn new(cfg: SgdConfig, num_workers: usize) -> SharedSgdBuilder {
    SharedSgdBuilder{
      cfg:          cfg,
      num_workers:  num_workers,
      shared_bar:   Arc::new(SpinBarrier::new(num_workers)),
      shared_grad:  Arc::new(Mutex::new(Vec::with_capacity(1024))),
    }
  }

  pub fn into_worker<S, Loss>(self, worker_rank: usize, operator: Rc<RefCell<Loss>>) -> SharedSgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
    let grad_sz = operator.borrow_mut().diff_param_sz();
    if worker_rank == 0 {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.resize(grad_sz, 0.0);
    }
    let cache = Vec::with_capacity(self.cfg.minibatch_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    let mut checkpoint = CheckpointState::default();
    if let Some(ref chk_cfg) = self.cfg.checkpoint {
      checkpoint = chk_cfg.build_state();
    }
    if let Some(ref mut config_file) = checkpoint.config_file {
      writeln!(config_file, "{:?}", self.cfg).unwrap();
    }
    let worker = SharedSgdWorker{
      cfg:          self.cfg,
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      checkpoint:   checkpoint,
      iter_counter: 0,
      operator:     operator,
      cache:        cache,
      grad_sz:      grad_sz,
      shared_bar:   self.shared_bar,
      shared_grad:  self.shared_grad,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      //_marker:      PhantomData,
    };
    worker.shared_bar.wait();
    worker
  }
}

pub struct SharedSgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  cfg:          SgdConfig,
  worker_rank:  usize,
  num_workers:  usize,
  checkpoint:   CheckpointState,
  iter_counter: usize,
  operator:     Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  grad_sz:      usize,
  shared_bar:   Arc<SpinBarrier>,
  shared_grad:  Arc<Mutex<Vec<f32>>>,
  param:        Vec<f32>,
  param_saved:  Vec<f32>,
  grad:         Vec<f32>,
  grad_acc:     Vec<f32>,
  stats_it:     usize,
  stats:        ClassOptStats,
  //_marker:      PhantomData<R>,
}

impl<S, Loss> SharedSgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
}

impl<S, Loss> OptWorker<f32, S> for SharedSgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Rng = Xorshiftplus128Rng;

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut operator = self.operator.borrow_mut();
    if self.worker_rank == 0 {
      operator.init_param(rng);
      operator.store_diff_param(&mut self.param_saved);
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.copy_from_slice(&self.param_saved);
    }
    self.shared_bar.wait();
    if self.worker_rank != 0 {
      let shared_grad = self.shared_grad.lock().unwrap();
      self.param_saved.copy_from_slice(&shared_grad);
      operator.load_diff_param(&mut self.param_saved);
    }
    self.shared_bar.wait();
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
      //sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
      self.cache.push(sample);
    }
    assert_eq!(self.cfg.minibatch_sz, self.cache.len());

    let mut operator = self.operator.borrow_mut();

    operator.save_rng_state();
    operator.reset_loss();
    operator.reset_grad();
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      //operator.store_diff_param(&mut self.param);
      self.param.copy_from_slice(&self.param_saved);
      self.param.reshape_mut(self.grad_sz).add(mu, self.grad_acc.reshape(self.grad_sz));
      operator.load_diff_param(&mut self.param);
    } else {
      operator.load_diff_param(&mut self.param_saved);
    }
    operator.next_iteration();
    for batch in 0 .. num_batches {
      let batch_start = batch * self.cfg.batch_sz;
      let batch_end = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz);
      operator.load_batch(&self.cache[batch_start .. batch_end]);
      operator.forward(OpPhase::Learning);
      operator.backward();
    }
    operator.update_nondiff_param(self.iter_counter);
    /*if let Some(GradientMomentum::Nesterov(_)) = self.cfg.momentum {
      operator.load_diff_param(&mut self.param_saved);
    }*/

    operator.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / (self.cfg.minibatch_sz * self.num_workers) as f32);
    let loss = operator.store_loss() / self.cfg.minibatch_sz as f32;

    if self.worker_rank == 0 {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.shared_bar.wait();
    {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.reshape_mut(self.grad_sz).vector_add(1.0, self.grad.reshape(self.grad_sz));
    }
    self.shared_bar.wait();
    {
      let shared_grad = self.shared_grad.lock().unwrap();
      self.grad.copy_from_slice(&shared_grad);
    }

    if let Some(GradientMomentum::HeavyBall(mu)) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.grad_sz).scale(mu);
    } else if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.grad_sz).scale(mu);
    } else {
      self.grad_acc.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.grad_acc.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));

    //operator.store_diff_param(&mut self.param);
    self.param.copy_from_slice(&self.param_saved);
    self.param.reshape_mut(self.grad_sz).add(1.0, self.grad_acc.reshape(self.grad_sz));
    //operator.load_diff_param(&mut self.param);
    self.param_saved.copy_from_slice(&self.param);

    self.iter_counter += 1;

    if let Some(ref mut train_file) = self.checkpoint.train_file {
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
    operator.load_diff_param(&mut self.param_saved);
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

impl<S, Loss> OptStats<ClassOptStats> for SharedSgdWorker<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> {
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
