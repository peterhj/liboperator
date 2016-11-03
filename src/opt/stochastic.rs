use prelude::*;

use densearray::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub trait StochasticUpdateStep<Loss, S> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Cfg: Clone;

  fn initialize(cfg: Self::Cfg, loss: &mut Loss) -> Self where Self: Sized;
  fn pre_step(&mut self, loss: &mut Loss);
  fn step(&mut self, minibatch_sz: usize, iter_count: usize, loss: &mut Loss, param_saved: &mut [f32]);
}

pub struct StochasticOptimizer<Update, Loss, S> where Update: StochasticUpdateStep<Loss, S>, Loss: DiffLoss<S, IoBuf=[f32]> {
  batch_sz:     usize,
  minibatch_sz: usize,
  grad_sz:      usize,
  cfg:          Update::Cfg,
  iter_count:   usize,
  loss:         Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  update_step:  Update,
  param_saved:  Vec<f32>,
  dirty_param:  bool,
}

impl<Update, Loss, S> StochasticOptimizer<Update, Loss, S> where Update: StochasticUpdateStep<Loss, S>, Loss: DiffLoss<S, IoBuf=[f32]> {
  pub fn new(batch_sz: usize, minibatch_sz: usize, cfg: Update::Cfg, loss: Rc<RefCell<Loss>>) -> StochasticOptimizer<Update, Loss, S> {
    let grad_sz = loss.borrow_mut().diff_param_sz();
    let cache = Vec::with_capacity(minibatch_sz);
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    StochasticOptimizer{
      batch_sz:     batch_sz,
      minibatch_sz: minibatch_sz,
      grad_sz:      grad_sz,
      cfg:          cfg.clone(),
      iter_count:   0,
      loss:         loss.clone(),
      cache:        cache,
      update_step:  StochasticUpdateStep::initialize(cfg, &mut *loss.borrow_mut()),
      param_saved:  param_saved,
      dirty_param:  true,
    }
  }

  pub fn init(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut loss = self.loss.borrow_mut();
    loss.init_param(rng);
    loss.store_diff_param(&mut self.param_saved);
    self.dirty_param = false;
  }

  pub fn step(&mut self, samples: &mut Iterator<Item=S>) {
    self.cache.clear();
    for mut sample in samples.take(self.minibatch_sz) {
      self.cache.push(sample);
    }
    assert_eq!(self.minibatch_sz, self.cache.len());

    let mut loss = self.loss.borrow_mut();
    loss.save_rng_state();
    loss.next_iteration();
    loss.reset_loss();
    loss.reset_grad();
    self.update_step.pre_step(&mut *loss);
    let num_batches = (self.minibatch_sz + self.batch_sz - 1) / self.batch_sz;
    for batch in 0 .. num_batches {
      let batch_start = batch * self.batch_sz;
      let batch_end = min((batch+1) * self.batch_sz, self.minibatch_sz);
      loss.load_batch(&self.cache[batch_start .. batch_end]);
      loss.forward(OpPhase::Learning);
      loss.backward();
    }
    self.update_step.step(self.minibatch_sz, self.iter_count, &mut *loss, &mut self.param_saved);

    self.iter_count += 1;
    self.cache.clear();
    self.dirty_param = true;
  }

  fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    let mut loss = self.loss.borrow_mut();
    loss.reset_loss();
    if self.dirty_param {
      loss.load_diff_param(&mut self.param_saved);
      self.dirty_param = false;
    }
    self.cache.clear();
    for mut sample in samples.take(epoch_sz) {
      self.cache.push(sample);
      if self.cache.len() == self.batch_sz {
        loss.load_batch(&self.cache);
        loss.forward(OpPhase::Inference);
        self.cache.clear();
      }
    }
    if self.cache.len() > 0 {
      loss.load_batch(&self.cache);
      loss.forward(OpPhase::Inference);
      self.cache.clear();
    }
  }
}

/*impl<S, Loss> StochasticOptimizer<S, Loss> where Loss: DiffLoss<S, IoBuf=[f32]> + LossReport<ClassLossStats> {
  pub fn update_stats(&self, stats: &mut ClassLossStats) {
    let mut operator = self.operator.borrow_mut();
    operator.update_stats(self.iter_count, stats);
  }
}*/
