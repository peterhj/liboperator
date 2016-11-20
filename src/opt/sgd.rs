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
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  diff_acc:     Vec<T>,
  _marker:      PhantomData<fn (Loss, S)>,
}

impl<Loss, S> GradUpdateStep<f32, Loss, S> for SgdUpdateStep<f32, Loss, S> where Loss: DiffLoss<S, IoBuf=[f32]> {
  type Cfg = SgdConfig;

  fn initialize(cfg: SgdConfig, loss: &mut Loss) -> SgdUpdateStep<f32, Loss, S> {
    let grad_sz = loss.diff_param_sz();
    let mut param = Vec::with_capacity(grad_sz);
    param.resize(grad_sz, 0.0);
    let mut param_saved = Vec::with_capacity(grad_sz);
    param_saved.resize(grad_sz, 0.0);
    let mut grad = Vec::with_capacity(grad_sz);
    grad.resize(grad_sz, 0.0);
    let mut diff_acc = Vec::with_capacity(grad_sz);
    diff_acc.resize(grad_sz, 0.0);
    SgdUpdateStep{
      cfg:          cfg,
      grad_sz:      grad_sz,
      param:        param,
      param_saved:  param_saved,
      grad:         grad,
      diff_acc:     diff_acc,
      _marker:      PhantomData,
    }
  }

  fn begin_iteration(&mut self, loss: &mut Loss) {
    if let Some(GradientMomentum::Nesterov(mu)) = self.cfg.momentum {
      loss.store_diff_param(&mut self.param_saved);
      self.param.copy_from_slice(&self.param_saved);
      self.param.reshape_mut(self.grad_sz).add(mu, self.diff_acc.reshape(self.grad_sz));
      loss.load_diff_param(&mut self.param);
    }
  }

  fn end_iteration(&mut self, minibatch_sz: usize, loss: &mut Loss) {
    loss.load_diff_param(&mut self.param_saved);
    loss.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);
  }

  fn step(&mut self, iter_count: usize, loss: &mut Loss) {
    /*loss.store_grad(&mut self.grad);
    self.grad.reshape_mut(self.grad_sz).div_scalar(minibatch_sz as f32);*/

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
