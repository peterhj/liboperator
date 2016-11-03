use prelude::*;
//use opt::stochastic::{StochasticUpdateStep};

use densearray::prelude::*;

#[derive(Clone, Debug)]
pub struct SgdConfig {
  pub step_size:    StepSize,
  pub momentum:     Option<GradientMomentum>,
}

pub struct SgdUpdateStep {
  minibatch_sz: usize,
  cfg:          SgdConfig,
  grad_sz:      usize,
  iter_count:   usize,
  param:        Vec<f32>,
  grad:         Vec<f32>,
  diff_acc:     Vec<f32>,
}

impl StochasticUpdateStep for SgdUpdateStep {
  type Cfg = SgdConfig;

  fn initialize<Loss, S>(minibatch_sz: usize, cfg: SgdConfig, loss: &mut Loss) -> SgdUpdateStep where Loss: DiffLoss<S, IoBuf=[f32]> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut diff_acc = Vec::with_capacity(grad_sz);
    diff_acc.resize(grad_sz, 0.0);
    SgdUpdateStep{
      minibatch_sz: minibatch_sz,
      cfg:          cfg,
      grad_sz:      grad_sz,
      iter_count:   0,
      param:        param,
      grad:         grad,
      diff_acc:     diff_acc,
    }
  }

  fn pre_step<Loss, S>(&mut self, loss: &mut Loss, param_saved: &mut [f32]) where Loss: DiffLoss<S, IoBuf=[f32]> {
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.param.copy_from_slice(param_saved);
      self.param.reshape_mut(self.grad_sz).add(mu, self.diff_acc.reshape(self.grad_sz));
      loss.load_diff_param(&mut self.param);
    } else {
      loss.load_diff_param(param_saved);
    }
  }

  fn step<Loss, S>(&mut self, loss: &mut Loss, param_saved: &mut [f32]) where Loss: DiffLoss<S, IoBuf=[f32]> {
    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = self.iter_count / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      _ => unimplemented!(),
    };
    loss.update_nondiff_param(self.iter_count);
    loss.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).div_scalar(self.minibatch_sz as f32);
    if let Some(GradientMomentum::HeavyBall(mu)) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
    } else if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      self.diff_acc.reshape_mut(self.grad_sz).scale(mu);
    } else {
      self.diff_acc.reshape_mut(self.grad_sz).set_constant(0.0);
    }
    self.diff_acc.reshape_mut(self.grad_sz).add(-step_size, self.grad.reshape(self.grad_sz));
    self.param.copy_from_slice(param_saved);
    self.param.reshape_mut(self.grad_sz).add(1.0, self.diff_acc.reshape(self.grad_sz));
    param_saved.copy_from_slice(&self.param);
    self.iter_count += 1;
  }
}
