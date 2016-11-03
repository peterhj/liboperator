use prelude::*;

use densearray::prelude::*;

use std::marker::{PhantomData};

#[derive(Clone, Debug)]
pub struct AdamConfig {
  pub step_size:    StepSize,
  pub gamma1:       f32,
  pub gamma2:       f32,
  pub epsilon:      f32,
}

pub struct AdamUpdateStep<Loss, S> {
  cfg:          AdamConfig,
  grad_sz:      usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  grad_acc:     Vec<f32>,
  grad_var_acc: Vec<f32>,
  //diff_acc:     Vec<f32>,
  tmp_buf:      Vec<f32>,
  _marker:      PhantomData<fn (Loss, S)>,
}

impl<Loss, S> StochasticUpdateStep<Loss, S> for AdamUpdateStep<Loss, S> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Cfg = AdamConfig;

  fn initialize(cfg: AdamConfig, loss: &mut Loss) -> AdamUpdateStep<Loss, S> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut grad_acc = Vec::with_capacity(grad_sz);
    grad_acc.resize(grad_sz, 0.0);
    let mut grad_var_acc = Vec::with_capacity(grad_sz);
    grad_var_acc.resize(grad_sz, 0.0);
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    tmp_buf.resize(grad_sz, 0.0);
    AdamUpdateStep{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      grad:         grad,
      grad_acc:     grad_acc,
      grad_var_acc: grad_var_acc,
      tmp_buf:      tmp_buf,
      _marker:      PhantomData,
    }
  }

  fn pre_step(&mut self, loss: &mut Loss) {
    loss.load_diff_param(&mut self.param);
  }

  fn step(&mut self, minibatch_sz: usize, iter_count: usize, loss: &mut Loss) {
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
    loss.update_nondiff_param(iter_count);
    loss.store_grad(&mut self.grad);
    let gamma1_scale = 1.0 / (1.0 - (1.0 - self.cfg.gamma1).powi((iter_count + 1) as i32));
    let gamma2_scale = 1.0 / (1.0 - (1.0 - self.cfg.gamma2).powi((iter_count + 1) as i32));
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);
    self.grad_acc.reshape_mut(self.grad_sz).average(self.cfg.gamma1, self.grad.reshape(self.grad_sz));
    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).average(self.cfg.gamma2, self.tmp_buf.reshape(self.grad_sz));
    self.tmp_buf.copy_from_slice(&self.grad_var_acc);
    self.tmp_buf.reshape_mut(self.grad_sz).scale(gamma2_scale);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_ldiv(self.grad_acc.reshape(self.grad_sz));
    self.tmp_buf.reshape_mut(self.grad_sz).scale(-step_size * gamma1_scale);
    self.param.reshape_mut(self.grad_sz).add(1.0, self.tmp_buf.reshape(self.grad_sz));
  }

  fn load_param(&mut self, src_param: &mut [f32]) {
    self.param.copy_from_slice(src_param);
  }

  fn save_param(&mut self, dst_param: &mut [f32]) {
    dst_param.copy_from_slice(&self.param);
  }
}
