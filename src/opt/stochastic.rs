use prelude::*;

use densearray::prelude::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cell::{RefCell};
use std::cmp::{min};
use std::marker::{PhantomData};
use std::num::{Zero};
use std::rc::{Rc};

pub trait GradUpdate<T, Loss, S, IoBuf: ?Sized> where T: Copy, Loss: DiffLoss<S, IoBuf> {
  type Cfg: Clone;

  fn initialize(cfg: Self::Cfg, loss: &mut Loss) -> Self where Self: Sized { unimplemented!(); }
  fn reset(&mut self, loss: &mut Loss, rng: &mut Xorshiftplus128Rng) { unimplemented!(); }
  fn begin_iteration(&mut self, loss: &mut Loss);
  fn end_iteration(&mut self, minibatch_sz: usize, loss: &mut Loss);
  fn step(&mut self, iter_count: usize, loss: &mut Loss);
  fn sync(&mut self) {}

  //fn pre_step(&mut self, loss: &mut Loss);
  //fn accumulate(&mut self, minibatch_sz: usize, loss: &mut Loss) { unimplemented!(); }
  //fn step(&mut self, minibatch_sz: usize, iter_count: usize, loss: &mut Loss);
  // FIXME(20161120): no point to saving/loading the temporary parameter.
  fn upload_param(&mut self, loss: &mut Loss) { unimplemented!(); }
  fn download_param(&mut self, loss: &mut Loss) { unimplemented!(); }
  fn load_param(&mut self, src_param: &mut [T]) { unimplemented!(); }
  fn save_param(&mut self, dst_param: &mut [T]) { unimplemented!(); }
}

pub struct StochasticGradWorker<T, Update, Loss, S, IoBuf: ?Sized> where T: Copy, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  batch_sz:     usize,
  minibatch_sz: usize,
  grad_sz:      usize,
  //cfg:          Update::Cfg,
  iter_count:   usize,
  loss:         Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  update:       Update,
  //param_saved:  Vec<T>,
  //dirty_param:  bool,
  stopwatch:    Stopwatch,
  _marker:      PhantomData<(T, fn (IoBuf))>,
}

impl<T, Update, Loss, S, IoBuf: ?Sized> StochasticGradWorker<T, Update, Loss, S, IoBuf> where T: Copy + Zero, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  pub fn new(batch_sz: usize, minibatch_sz: usize, /*cfg: Update::Cfg,*/ update: Update, loss: Rc<RefCell<Loss>>) -> StochasticGradWorker<T, Update, Loss, S, IoBuf> {
    let grad_sz = loss.borrow_mut().diff_param_sz();
    let cache = Vec::with_capacity(minibatch_sz);
    //let mut param_saved = Vec::with_capacity(grad_sz);
    //param_saved.resize(grad_sz, T::zero());
    StochasticGradWorker{
      batch_sz:     batch_sz,
      minibatch_sz: minibatch_sz,
      grad_sz:      grad_sz,
      //cfg:          cfg.clone(),
      iter_count:   0,
      loss:         loss.clone(),
      cache:        cache,
      update:       update, //GradUpdate::initialize(cfg, &mut *loss.borrow_mut()),
      //param_saved:  param_saved,
      //dirty_param:  true,
      stopwatch:    Stopwatch::new(),
      _marker:      PhantomData,
    }
  }
}

impl<T, Update, Loss, S, IoBuf: ?Sized> StochasticGradWorker<T, Update, Loss, S, IoBuf> where T: Copy, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  pub fn init(&mut self, rng: &mut Xorshiftplus128Rng) {
    //let mut loss = self.loss.borrow_mut();
    //loss.init_param(rng);

    self.stopwatch.lap();
    self.update.reset(&mut *self.loss.borrow_mut(), rng);
    self.stopwatch.lap();
    //println!("DEBUG: sg: init: {:.6}", self.stopwatch.elapsed());
  }

  pub fn step(&mut self, samples: &mut Iterator<Item=S>) {
    self.stopwatch.lap();
    self.cache.clear();
    for sample in samples.take(self.minibatch_sz) {
      self.cache.push(sample);
    }
    assert_eq!(self.minibatch_sz, self.cache.len());
    self.stopwatch.lap();
    //println!("DEBUG: sg: step: fetching samples: {:.6}", self.stopwatch.elapsed());

    let mut loss = self.loss.borrow_mut();
    self.update.begin_iteration(&mut *loss);
    loss.save_rng_state();
    loss.next_iteration();
    loss.reset_loss();
    loss.reset_grad();
    let num_batches = (self.minibatch_sz + self.batch_sz - 1) / self.batch_sz;
    for batch in 0 .. num_batches {
      let batch_start = batch * self.batch_sz;
      let batch_end = min((batch + 1) * self.batch_sz, self.minibatch_sz);
      self.stopwatch.lap();
      loss.load_batch(&self.cache[batch_start .. batch_end]);
      self.stopwatch.lap();
      //println!("DEBUG: sg: step: loading batch: {:.6}", self.stopwatch.elapsed());
      loss.forward(OpPhase::Learning);
      self.stopwatch.lap();
      //println!("DEBUG: sg: step: forward: {:.6}", self.stopwatch.elapsed());
      loss.backward();
      self.stopwatch.lap();
      //println!("DEBUG: sg: step: backward: {:.6}", self.stopwatch.elapsed());
    }
    self.update.end_iteration(self.minibatch_sz, &mut *loss);
    self.update.step(self.iter_count, &mut *loss);
    loss.update_nondiff_param(self.iter_count);
    self.stopwatch.lap();
    //println!("DEBUG: sg: step: update: {:.6}", self.stopwatch.elapsed());

    self.iter_count += 1;
    self.cache.clear();
    //self.dirty_param = true;
  }

  pub fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    let mut loss = self.loss.borrow_mut();
    loss.reset_loss();
    /*if self.dirty_param {
      //self.update.save_param(&mut self.param_saved);
      //loss.load_diff_param(&mut self.param_saved);
      loss.store_diff_param(&mut self.param_saved);
      self.dirty_param = false;
    }*/
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

impl<Update, Loss, S, IoBuf: ?Sized> StochasticGradWorker<f32, Update, Loss, S, IoBuf> where Update: GradUpdate<f32, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> + LossReport<ClassLossStats> {
  pub fn update_stats(&self, stats: &mut ClassLossStats) {
    let mut operator = self.loss.borrow_mut();
    operator.update_stats(self.iter_count, stats);
  }
}

pub struct FastStochasticGradWorker<T, Update, Loss, S, IoBuf: ?Sized> where T: Copy, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  minibatch_sz: usize,
  grad_sz:      usize,
  iter_count:   usize,
  loss:         Rc<RefCell<Loss>>,
  cache:        Vec<S>,
  update:  Update,
  stopwatch:    Stopwatch,
  _marker:      PhantomData<(T, fn (IoBuf))>,
}

impl<T, Update, Loss, S, IoBuf: ?Sized> FastStochasticGradWorker<T, Update, Loss, S, IoBuf> where T: Copy + Zero, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  pub fn new(minibatch_sz: usize, /*cfg: Update::Cfg,*/ update: Update, loss: Rc<RefCell<Loss>>) -> FastStochasticGradWorker<T, Update, Loss, S, IoBuf> {
    let grad_sz = loss.borrow_mut().diff_param_sz();
    let cache = Vec::with_capacity(minibatch_sz);
    FastStochasticGradWorker{
      minibatch_sz: minibatch_sz,
      grad_sz:      grad_sz,
      iter_count:   0,
      loss:         loss.clone(),
      cache:        cache,
      update:  update,
      stopwatch:    Stopwatch::new(),
      _marker:      PhantomData,
    }
  }
}

impl<T, Update, Loss, S, IoBuf: ?Sized> FastStochasticGradWorker<T, Update, Loss, S, IoBuf> where T: Copy, Update: GradUpdate<T, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> {
  pub fn init(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.stopwatch.lap();
    self.update.reset(&mut *self.loss.borrow_mut(), rng);
    self.stopwatch.lap();
    println!("DEBUG: sg: init: {:.6}", self.stopwatch.elapsed());
  }

  pub fn step(&mut self, samples: &mut Iterator<Item=S>) {
    self.stopwatch.lap();
    self.cache.clear();
    for sample in samples.take(self.minibatch_sz) {
      self.cache.push(sample);
    }
    assert_eq!(self.minibatch_sz, self.cache.len());
    self.stopwatch.lap();
    println!("DEBUG: sg: step: fetching samples: {:.6}", self.stopwatch.elapsed());

    let mut loss = self.loss.borrow_mut();
    self.update.begin_iteration(&mut *loss);
    loss.save_rng_state();
    loss.next_iteration();
    loss.reset_loss();
    loss.reset_grad();
    {
      self.stopwatch.lap();
      loss.load_batch(&self.cache);
      self.stopwatch.lap();
      println!("DEBUG: sg: step: loading batch: {:.6}", self.stopwatch.elapsed());
      loss.forward(OpPhase::Learning);
      self.stopwatch.lap();
      println!("DEBUG: sg: step: forward: {:.6}", self.stopwatch.elapsed());
    }
    self.update.end_iteration(self.minibatch_sz, &mut *loss);
    self.update.step(self.iter_count, &mut *loss);
    loss.update_nondiff_param(self.iter_count);
    self.stopwatch.lap();
    println!("DEBUG: sg: step: backward + update: {:.6}", self.stopwatch.elapsed());

    self.iter_count += 1;
    self.cache.clear();
  }

  pub fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    let mut loss = self.loss.borrow_mut();
    loss.reset_loss();
    self.cache.clear();
    for mut sample in samples.take(epoch_sz) {
      self.cache.push(sample);
      if self.cache.len() == self.minibatch_sz {
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

  pub fn sync(&mut self) {
    self.update.sync();
  }
}

impl<Update, Loss, S, IoBuf: ?Sized> FastStochasticGradWorker<f32, Update, Loss, S, IoBuf> where Update: GradUpdate<f32, Loss, S, IoBuf>, Loss: DiffLoss<S, IoBuf> + LossReport<ClassLossStats> {
  pub fn update_stats(&self, stats: &mut ClassLossStats) {
    let mut operator = self.loss.borrow_mut();
    operator.update_stats(self.iter_count, stats);
  }
}
