use prelude::*;

use densearray::prelude::*;

use std::marker::{PhantomData};

#[derive(Clone, Debug)]
pub struct SgdConfig {
  pub step_size:    StepSize,
  pub momentum:     Option<GradientMomentum>,
}

pub struct SgdUpdateStep<T, Loss, S> where T: Copy {
  cfg:          SgdConfig,
  grad_sz:      usize,
  param:        Vec<T>,
  param_mirror: Vec<T>,
  grad:         Vec<T>,
  diff_acc:     Vec<T>,
  _marker:      PhantomData<fn (Loss, S)>,
}

impl<Loss, S> StochasticUpdateStep<f32, Loss, S> for SgdUpdateStep<f32, Loss, S> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Cfg = SgdConfig;

  fn initialize(cfg: SgdConfig, loss: &mut Loss) -> SgdUpdateStep<f32, Loss, S> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut param_mirror = Vec::with_capacity(grad_sz);
    param_mirror.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut diff_acc = Vec::with_capacity(grad_sz);
    diff_acc.resize(grad_sz, 0.0);
    SgdUpdateStep{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      param_mirror: param_mirror,
      grad:         grad,
      diff_acc:     diff_acc,
      _marker:      PhantomData,
    }
  }

  fn pre_step(&mut self, loss: &mut Loss) {
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.param_mirror.copy_from_slice(&self.param);
      self.param_mirror.reshape_mut(self.grad_sz).add(mu, self.diff_acc.reshape(self.grad_sz));
      loss.load_diff_param(&mut self.param_mirror);
    } else {
      loss.load_diff_param(&mut self.param);
    }
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
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);
    if let Some(GradientMomentum::HeavyBall(mu)) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
      self.diff_acc.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));
      self.param.reshape_mut(self.grad_sz).add(1.0, self.diff_acc.reshape(self.grad_sz));
    } else if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
      self.diff_acc.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));
      self.param.reshape_mut(self.grad_sz).add(1.0, self.diff_acc.reshape(self.grad_sz));
    } else {
      self.param.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));
    }
  }

  fn load_param(&mut self, src_param: &mut [f32]) {
    self.param.copy_from_slice(src_param);
  }

  fn save_param(&mut self, dst_param: &mut [f32]) {
    dst_param.copy_from_slice(&self.param);
  }
}
