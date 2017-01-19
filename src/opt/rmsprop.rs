use prelude::*;

use densearray::prelude::*;
use rng::xorshift::*;

use std::cell::{RefCell};
use std::cmp::{min};
use std::rc::{Rc};

#[derive(Clone, Debug)]
pub struct RmspropConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub rms_decay:    f32,
  pub momentum:     Option<f32>,
  pub epsilon:      f32,
}

pub struct RmspropUpdate<T> where T: Copy {
  cfg:          RmspropConfig,
  grad_sz:      usize,
  param:        Vec<T>,
  grad:         Vec<T>,
  grad_var_acc: Vec<T>,
  diff_acc:     Vec<T>,
  tmp_buf:      Vec<T>,
}

impl RmspropUpdate<f32> {
  pub fn new<Loss, S>(cfg: RmspropConfig, loss: &mut Loss) -> RmspropUpdate<f32>
  where Loss: DiffLoss<S, [f32]> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut diff_acc = Vec::with_capacity(grad_sz);
    diff_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    RmspropUpdate{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      grad:         grad,
      grad_var_acc: grad_var_acc,
      diff_acc:     diff_acc,
      tmp_buf:      tmp_buf,
    }
  }
}

impl<Loss, S> GradUpdate<f32, Loss, S, [f32]> for RmspropUpdate<f32> where Loss: DiffLoss<S, [f32]> {
  type Cfg = RmspropConfig;

  fn initialize(cfg: RmspropConfig, loss: &mut Loss) -> RmspropUpdate<f32> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut diff_acc = Vec::with_capacity(grad_sz);
    diff_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    RmspropUpdate{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      grad:         grad,
      grad_var_acc: grad_var_acc,
      diff_acc:     diff_acc,
      tmp_buf:      tmp_buf,
    }
  }

  fn begin_iteration(&mut self, loss: &mut Loss) {
  }

  fn end_iteration(&mut self, minibatch_sz: usize, loss: &mut Loss) {
    loss.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);
  }

  fn step(&mut self, iter_count: usize, loss: &mut Loss) {
    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).average(1.0 - self.cfg.rms_decay, self.tmp_buf.reshape(self.grad_sz));
    let rms_decay_scale = 1.0 / (1.0 - self.cfg.rms_decay.powi((iter_count + 1) as i32));
    self.tmp_buf.copy_from_slice(&self.grad_var_acc);
    self.tmp_buf.reshape_mut(self.grad_sz).scale(rms_decay_scale);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_ldiv(self.grad.reshape(self.grad_sz));
    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = iter_count / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      _ => unimplemented!(),
    };
    loss.store_diff_param(&mut self.param);
    if let Some(mu) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
      self.diff_acc.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
      self.param.reshape_mut(self.grad_sz).add(1.0, self.diff_acc.reshape(self.grad_sz));
    } else {
      self.param.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
    }
    loss.load_diff_param(&mut self.param);
  }

  fn download_param(&mut self, loss: &mut Loss) {
    loss.store_diff_param(&mut self.param);
  }

  fn upload_param(&mut self, loss: &mut Loss) {
    loss.load_diff_param(&mut self.param);
  }

  fn load_param(&mut self, src_param: &mut [f32]) {
    self.param.copy_from_slice(src_param);
  }

  fn save_param(&mut self, dst_param: &mut [f32]) {
    dst_param.copy_from_slice(&self.param);
  }
}

pub struct RmspropWorker<T, Loss, S> where T: Copy {
  cfg:          RmspropConfig,
  grad_sz:      usize,
  loss:         Rc<RefCell<Loss>>,
  samples:      Vec<S>,
  iter_count:   usize,
  param:        Vec<T>,
  grad:         Vec<T>,
  grad_var_acc: Vec<T>,
  step_acc:     Vec<T>,
  tmp_buf:      Vec<T>,
  //tmp_buf2:     Vec<T>,
}

impl<Loss, S> RmspropWorker<f32, Loss, S> where Loss: DiffLoss<S, [f32]> {
  pub fn new(cfg: RmspropConfig, loss: Rc<RefCell<Loss>>) -> RmspropWorker<f32, Loss, S> {
    let grad_sz = loss.borrow_mut().diff_param_sz();
    let samples = Vec::with_capacity(cfg.minibatch_sz);
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut step_acc = Vec::with_capacity(grad_sz);
    step_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    let mut tmp_buf2 = Vec::with_capacity(grad_sz);
    tmp_buf2.resize(grad_sz, 0.0);
    RmspropWorker{
      cfg:          cfg,
      grad_sz:      grad_sz,
      loss:         loss,
      samples:      samples,
      iter_count:   0,
      param:        param,
      grad:         grad,
      grad_var_acc: grad_var_acc,
      step_acc:     step_acc,
      tmp_buf:      tmp_buf,
      //tmp_buf2:     tmp_buf2,
    }
  }

  pub fn reset(&mut self, rng: &mut Xorshiftplus128Rng) {
    let mut loss = self.loss.borrow_mut();
    loss.init_param(rng);
    //loss.store_diff_param(&mut self.param);
  }

  pub fn step(&mut self, samples_iter: &mut Iterator<Item=S>) {
    let mut loss = self.loss.borrow_mut();
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    let minibatch_sz = self.cfg.minibatch_sz;
    let step_size = self.cfg.step_size.at_iter(self.iter_count);

    self.samples.clear();
    for sample in samples_iter.take(self.cfg.minibatch_sz) {
      self.samples.push(sample);
    }
    assert_eq!(self.cfg.minibatch_sz, self.samples.len());

    //loss.save_rng_state();
    loss.next_iteration();
    loss.reset_loss();
    loss.reset_grad();
    for batch in 0 .. num_batches {
      let batch_start = self.cfg.batch_sz * batch;
      let batch_end = min(self.cfg.minibatch_sz, self.cfg.batch_sz * (batch + 1));
      loss.load_batch(&self.samples[batch_start .. batch_end]);
      loss.forward(OpPhase::Learning);
      loss.backward();
    }
    loss.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).scale(1.0 / minibatch_sz as f32);
    loss.update_nondiff_param(self.iter_count);

    let gamma = self.cfg.rms_decay;
    let gamma_scale = 1.0 / (1.0 - (1.0 - gamma).powi((self.iter_count + 1) as i32));
    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).scale(1.0 - gamma);
    self.grad_var_acc.reshape_mut(self.grad_sz).add(gamma, self.tmp_buf.reshape(self.grad_sz));
    self.tmp_buf.copy_from_slice(&self.grad_var_acc);
    self.tmp_buf.reshape_mut(self.grad_sz).scale(gamma_scale);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).reciprocal();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_mult(self.grad.reshape(self.grad_sz));

    loss.store_diff_param(&mut self.param);
    if let Some(mu) = self.cfg.momentum {
      self.step_acc.reshape_mut(self.grad_sz).scale(mu);
      self.step_acc.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
      self.param.reshape_mut(self.grad_sz).add(1.0, self.step_acc.reshape(self.grad_sz));
    } else {
      self.param.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
    }
    loss.load_diff_param(&mut self.param);
  }

  pub fn eval(&mut self, epoch_sz: usize, samples_iter: &mut Iterator<Item=S>) {
    let mut loss = self.loss.borrow_mut();
    loss.reset_loss();
    self.samples.clear();
    for mut sample in samples_iter.take(epoch_sz) {
      self.samples.push(sample);
      assert!(self.samples.len() <= self.cfg.batch_sz);
      if self.samples.len() == self.cfg.batch_sz {
        loss.load_batch(&self.samples);
        loss.forward(OpPhase::Inference);
        self.samples.clear();
      }
    }
    if self.samples.len() > 0 {
      loss.load_batch(&self.samples);
      loss.forward(OpPhase::Inference);
      self.samples.clear();
    }
  }
}

impl<Loss, S> RmspropWorker<f32, Loss, S> where Loss: DiffLoss<S, [f32]> + LossReport<ClassLossStats> {
  pub fn update_stats(&self, stats: &mut ClassLossStats) {
    let mut loss = self.loss.borrow_mut();
    loss.update_stats(self.iter_count, stats);
  }
}
