use prelude::*;
use data::{SampleWeight};
use opt::{
  ClassOptStats,
};
use opt::sgd::{SgdConfig};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};
use sharedmem::{SharedMem};
use sharedmem::sync::{SpinBarrier};

use rand::{Rng};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SharedSyncSgdBuilder<T> {
  cfg:          SgdConfig,
  num_workers:  usize,
  shared_grad:  Arc<Mutex<Vec<T>>>,
  shared_bar:   Arc<SpinBarrier>,
}

impl<T> SharedSyncSgdBuilder<T> {
  pub fn new(cfg: SgdConfig, num_workers: usize) -> SharedSyncSgdBuilder<T> {
    SharedSyncSgdBuilder{
      cfg:          cfg,
      num_workers:  num_workers,
      shared_grad:  Arc::new(Mutex::new(Vec::with_capacity(1024))),
      shared_bar:   Arc::new(SpinBarrier::new(num_workers)),
    }
  }
}

impl SharedSyncSgdBuilder<f32> {
  pub fn into_worker<S, R, Op>(self, worker_rank: usize, operator: Op) -> SharedSyncSgdWorker<f32, S, R, Op> where R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
    let grad_sz = operator.diff_param_sz();
    if worker_rank == 0 {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.resize(grad_sz, 0.0);
    }
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    let mut worker = SharedSyncSgdWorker{
      cfg:          self.cfg,
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      iter_counter: 0,
      operator:     operator,
      cache:        Vec::with_capacity(self.cfg.minibatch_sz),
      grad_sz:      grad_sz,
      shared_grad:  self.shared_grad,
      shared_bar:   self.shared_bar,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      _marker:      PhantomData,
    };
    worker.shared_bar.wait();
    worker
  }
}

pub struct SharedSyncSgdWorker<T, S, R, Op> where R: Rng, Op: DiffOperatorInput<T, S> {
  cfg:          SgdConfig,
  worker_rank:  usize,
  num_workers:  usize,
  iter_counter: usize,
  operator:     Op,
  cache:        Vec<S>,
  grad_sz:      usize,
  shared_grad:  Arc<Mutex<Vec<T>>>,
  shared_bar:   Arc<SpinBarrier>,
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  grad_acc:     Vec<T>,
  stats_it:     usize,
  stats:        ClassOptStats,
  _marker:      PhantomData<R>,
}

impl<S, R, Op> OptWorker<f32, S> for SharedSyncSgdWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
  type Rng = R;

  fn init_param(&mut self, rng: &mut Op::Rng) {
    if self.worker_rank == 0 {
      self.operator.init_param(rng);
      self.operator.store_param(&mut self.param_saved, 0);
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.copy_from_slice(&self.param_saved);
    }
    self.shared_bar.wait();
    if self.worker_rank != 0 {
      let shared_grad = self.shared_grad.lock().unwrap();
      self.param_saved.copy_from_slice(&shared_grad);
      self.operator.load_param(&mut self.param_saved, 0);
    }
    self.shared_bar.wait();
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
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
      sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
      self.cache.push(sample);
    }

    self.operator.save_rng_state();
    self.operator.reset_loss();
    self.operator.reset_grad();
    if let Some(mu) = self.cfg.momentum {
      self.operator.update_param(mu, 1.0, &mut self.grad_acc, 0);
    }
    for batch in 0 .. num_batches {
      let batch_start = batch * self.cfg.batch_sz;
      let batch_end = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz);
      self.operator.load_data(&self.cache[batch_start .. batch_end]);
      self.operator.forward(OpPhase::Learning);
      self.operator.backward();
    }
    if let Some(_) = self.cfg.momentum {
      self.operator.load_param(&mut self.param_saved, 0);
    }
    self.operator.store_grad(&mut self.grad, 0);
    let loss = self.operator.store_loss();

    if self.worker_rank == 0 {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.shared_bar.wait();
    {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.reshape_mut(self.grad_sz).vector_add(1.0 / self.num_workers as f32, self.grad.reshape(self.grad_sz));
    }
    self.shared_bar.wait();
    {
      let shared_grad = self.shared_grad.lock().unwrap();
      self.grad.copy_from_slice(&shared_grad);
    }

    if let Some(mu) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.grad_sz).vector_scale(mu);
    } else {
      self.grad_acc.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.grad_acc.reshape_mut(self.grad_sz).vector_add(-step_size, self.grad.reshape(self.grad_sz));

    self.operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
    self.operator.update_nondiff_param(self.iter_counter);
    self.operator.store_param(&mut self.param_saved, 0);

    self.iter_counter += 1;

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (loss - self.stats.avg_loss);
  }

  fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    self.cache.clear();
    self.operator.reset_loss();
    for mut sample in samples.take(epoch_sz) {
      sample.mix_weight(1.0 / epoch_sz as f32);
      self.cache.push(sample);
      if self.cache.len() == self.cfg.batch_sz {
        self.operator.load_data(&self.cache);
        self.operator.forward(OpPhase::Inference);
        self.cache.clear();
      }
    }
    if self.cache.len() > 0 {
      self.operator.load_data(&self.cache);
      self.operator.forward(OpPhase::Inference);
    }
    self.stats_it += 1;
    self.stats.sample_count += epoch_sz;
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (self.operator.store_loss() - self.stats.avg_loss);
  }
}

impl<S, R, Op> OptStats<ClassOptStats> for SharedSyncSgdWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S> {
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

#[derive(Clone)]
pub struct SharedAsyncSgdBuilder<T> {
  cfg:          SgdConfig,
  num_workers:  usize,
  //shared_grad:  Arc<Mutex<Vec<T>>>,
  shared_bar:   Arc<SpinBarrier>,
  shared_param: SharedMem<T>,
}

impl<T> SharedAsyncSgdBuilder<T> {
  pub fn new(cfg: SgdConfig, num_workers: usize) -> SharedAsyncSgdBuilder<T> {
    unimplemented!();
    /*SharedAsyncSgdBuilder{
      cfg:          cfg,
      num_workers:  num_workers,
      //shared_grad:  Arc::new(Mutex::new(Vec::with_capacity(1024))),
      shared_bar:   Arc::new(SpinBarrier::new(num_workers)),
    }*/
  }
}

impl SharedAsyncSgdBuilder<f32> {
  pub fn into_worker<S, R, Op>(self, worker_rank: usize, operator: Op) -> SharedAsyncSgdWorker<f32, S, R, Op> where R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
    unimplemented!();
    /*let grad_sz = operator.diff_param_sz();
    if worker_rank == 0 {
      let mut shared_grad = self.shared_grad.lock().unwrap();
      shared_grad.resize(grad_sz, 0.0);
    }
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    let mut worker = SharedAsyncSgdWorker{
      cfg:          self.cfg,
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      iter_counter: 0,
      operator:     operator,
      cache:        Vec::with_capacity(self.cfg.minibatch_sz),
      grad_sz:      grad_sz,
      shared_grad:  self.shared_grad,
      shared_bar:   self.shared_bar,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      _marker:      PhantomData,
    };
    worker.shared_bar.wait();
    worker*/
  }
}

pub struct SharedAsyncSgdWorker<T, S, R, Op> where R: Rng, Op: DiffOperatorInput<T, S> {
  cfg:          SgdConfig,
  iter_counter: usize,
  operator:     Op,
  cache:        Vec<S>,
  grad_sz:      usize,
  shared_param: SharedMem<T>,
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  grad_acc:     Vec<T>,
  stats_it:     usize,
  stats:        ClassOptStats,
  _marker:      PhantomData<R>,
}
