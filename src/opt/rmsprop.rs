use prelude::*;

use densearray::prelude::*;

use std::marker::{PhantomData};

#[derive(Clone, Debug)]
pub struct RmspropConfig {
  pub step_size:    StepSize,
  pub rms_decay:    f32,
  pub momentum:     Option<f32>,
  pub epsilon:      f32,
}

pub struct RmspropUpdateStep<Loss, S> {
  cfg:          RmspropConfig,
  grad_sz:      usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  grad_var_acc: Vec<f32>,
  diff_acc:     Vec<f32>,
  tmp_buf:      Vec<f32>,
  _marker:      PhantomData<(Loss, S)>,
}

impl<Loss, S> StochasticUpdateStep<Loss, S> for RmspropUpdateStep<Loss, S> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Cfg = RmspropConfig;

  fn initialize(cfg: RmspropConfig, loss: &mut Loss) -> RmspropUpdateStep<Loss, S> {
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
    RmspropUpdateStep{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      grad:         grad,
      grad_var_acc: grad_var_acc,
      diff_acc:     diff_acc,
      tmp_buf:      tmp_buf,
      _marker:      PhantomData,
    }
  }

  fn pre_step(&mut self, loss: &mut Loss) {
    loss.load_diff_param(&mut self.param);
  }

  fn step(&mut self, minibatch_sz: usize, iter_count: usize, loss: &mut Loss, param_saved: &mut [f32]) {
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
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);
    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).square();
    self.grad_var_acc.reshape_mut(self.grad_sz).average(1.0 - self.cfg.rms_decay, self.tmp_buf.reshape(self.grad_sz));
    self.tmp_buf.copy_from_slice(&self.grad_var_acc);
    self.tmp_buf.reshape_mut(self.grad_sz).add_scalar(self.cfg.epsilon);
    self.tmp_buf.reshape_mut(self.grad_sz).sqrt();
    self.tmp_buf.reshape_mut(self.grad_sz).elem_ldiv(self.grad.reshape(self.grad_sz));
    if let Some(mu) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
      self.diff_acc.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
      self.param.reshape_mut(self.grad_sz).add(1.0, self.diff_acc.reshape(self.grad_sz));
    } else {
      self.param.reshape_mut(self.grad_sz).add(-step_size, self.tmp_buf.reshape(self.grad_sz));
    }
    param_saved.copy_from_slice(&self.param);
  }
}
