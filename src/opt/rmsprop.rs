use prelude::*;
use data::{SampleWeight};
use opt::{GradientMomentum, NesterovParamState, ClassOptStats};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};

use rand::{Rng};
use std::cmp::{min};
use std::marker::{PhantomData};

#[derive(Clone, Copy, Debug)]
pub struct RmspropConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  //pub momentum:     Option<GradientMomentum>,
  pub momentum:     Option<f32>,
  pub gamma:        f32,
  /// `centered_var`: When enabled, this stores the 1st moment of the gradients
  /// and uses it to center the variance; c.f. [Graves 2013, arXiv:1308.0850].
  pub centered_var: bool,
  pub epsilon:      f32,
  pub l2_reg:       Option<f32>,
}

pub struct RmspropWorker<T, S, R, Op> where R: Rng, Op: DiffOperatorInput<T, S> {
  cfg:          RmspropConfig,
  iter_counter: usize,
  operator:     Op,
  cache:        Vec<S>,
  grad_sz:      usize,
  nondiff_sz:   usize,
  //param_state:  NesterovParamState,
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  grad_acc:     Option<Vec<T>>,
  sq_grad_acc:  Vec<T>,
  nrm_update:   Vec<T>,
  //nrm_upd_acc:  Vec<T>,
  tmp_buf:      Vec<T>,
  stats_it:     usize,
  stats:        ClassOptStats,
  _marker:      PhantomData<R>,
}

impl<S, R, Op> RmspropWorker<f32, S, R, Op> where R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
  pub fn new(cfg: RmspropConfig, operator: Op) -> RmspropWorker<f32, S, R, Op> {
    let grad_sz = operator.param_len();
    //let grad_sz = operator.diff_param_sz();
    let mut param_saved = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      param_saved.push(0.0);
    }
    let mut grad = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      grad.push(0.0);
    }
    let grad_acc = if cfg.centered_var {
    let mut grad_acc = Vec::with_capacity(grad_sz);
      for _ in 0 .. grad_sz {
        grad_acc.push(0.0);
      }
      Some(grad_acc)
    } else {
      None
    };
    let mut sq_grad_acc = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      sq_grad_acc.push(0.0);
    }
    let mut nrm_update = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      nrm_update.push(0.0);
    }
    let mut tmp_buf = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      tmp_buf.push(0.0);
    }
    RmspropWorker{
      cfg:          cfg,
      iter_counter: 0,
      operator:     operator,
      cache:        Vec::with_capacity(cfg.batch_sz),
      grad_sz:      grad_sz,
      nondiff_sz:   0, // FIXME(20161001): count nondiff params too (e.g. batchnorm).
      //param_state:  NesterovParamState::Orig,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      sq_grad_acc:  sq_grad_acc,
      nrm_update:   nrm_update,
      tmp_buf:      tmp_buf,
      stats_it:     0,
      stats:        Default::default(),
      _marker:      PhantomData,
    }
  }
}

impl<S, R, Op> OptWorker<f32, S> for RmspropWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
  type Rng = R;

  fn init_param(&mut self, rng: &mut Op::Rng) {
    self.operator.init_param(rng);
    self.operator.store_param(&mut self.param_saved, 0);
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn step(&mut self, samples: &mut Iterator<Item=S>) {
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;

    self.operator.save_rng_state();
    self.operator.reset_loss();
    self.operator.reset_grad();
    for batch in 0 .. num_batches {
      let actual_batch_sz = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
      self.cache.clear();
      for mut sample in samples.take(actual_batch_sz) {
        sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
        self.cache.push(sample);
      }
      self.operator.load_data(&self.cache);
      self.operator.forward(OpPhase::Learning);
      if let Some(lambda) = self.cfg.l2_reg {
        //self.operator.fwd_reg(Regularization::L2(lambda));
      }
      self.operator.backward();
      if let Some(lambda) = self.cfg.l2_reg {
        //self.operator.bwd_reg(Regularization::L2(lambda));
      }
    }
    if let Some(lambda) = self.cfg.l2_reg {
      self.operator.apply_grad_reg(Regularization::L2(lambda));
    }
    self.operator.store_grad(&mut self.grad, 0);

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

    if let Some(ref mut grad_acc) = self.grad_acc {
      if self.iter_counter == 0 {
        grad_acc.copy_from_slice(&self.grad);
      } else {
        grad_acc.reshape_mut(self.grad_sz).vector_scale(1.0 - self.cfg.gamma);
        grad_acc.reshape_mut(self.grad_sz).vector_add(self.cfg.gamma, self.grad.reshape(self.grad_sz));
      }
    }
    self.tmp_buf.copy_from_slice(&self.grad);
    self.tmp_buf.reshape_mut(self.grad_sz).vector_square();
    if self.iter_counter == 0 {
      self.sq_grad_acc.copy_from_slice(&self.tmp_buf);
    } else {
      self.sq_grad_acc.reshape_mut(self.grad_sz).vector_scale(1.0 - self.cfg.gamma);
      self.sq_grad_acc.reshape_mut(self.grad_sz).vector_add(self.cfg.gamma, self.tmp_buf.reshape(self.grad_sz));
    }

    self.nrm_update.copy_from_slice(&self.sq_grad_acc);
    if let Some(ref grad_acc) = self.grad_acc {
      self.tmp_buf.copy_from_slice(grad_acc);
      self.tmp_buf.reshape_mut(self.grad_sz).vector_square();
      self.nrm_update.reshape_mut(self.grad_sz).vector_add(-1.0, self.tmp_buf.reshape(self.grad_sz));
    }
    self.nrm_update.reshape_mut(self.grad_sz).vector_add_scalar(self.cfg.epsilon);
    self.nrm_update.reshape_mut(self.grad_sz).vector_sqrt();
    self.nrm_update.reshape_mut(self.grad_sz).vector_recip();
    self.nrm_update.reshape_mut(self.grad_sz).vector_elem_mult(1.0, self.grad.reshape(self.grad_sz));

    self.operator.update_param(-step_size, 1.0, &mut self.nrm_update, 0);
    self.operator.store_param(&mut self.param_saved, 0);

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (self.operator.store_loss() - self.stats.avg_loss);

    self.iter_counter += 1;
  }

  fn eval(&mut self, epoch_sz: usize, samples: &mut Iterator<Item=S>) {
    /*match self.param_state {
      NesterovParamState::Orig => {}
      NesterovParamState::PlusMomentum => {
        //self.operator.load_param(..., 0);
        self.param_state = NesterovParamState::Orig;
      }
    }*/
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

impl<S, R, Op> OptStats<ClassOptStats> for RmspropWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S> {
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
