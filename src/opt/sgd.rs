use prelude::*;
use data::{SampleWeight};
use opt::{
  ClassOptStats,
  //AdaptiveStepSizeState,
  adaptive_pow10_step_size_factor,
  adaptive_decimal_step_size_factor,
};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};

use rand::{Rng};
use std::cmp::{min};
use std::marker::{PhantomData};

struct SgdAdaptiveState {
  batch_sz:     usize,
  minibatch_sz: usize,
  momentum:     Option<f32>,
  //momentum:     Option<GradientMomentum>,
  grad_sz:      usize,
  init_step:    f32,
  test_iters:   usize,
  epoch_iters:  usize,
  sched:        AdaptiveStepSizeSchedule,
  prev_step:    f32,
  param_sav:    Vec<f32>,
  grad:         Vec<f32>,
  grad_acc:     Vec<f32>,
}

impl SgdAdaptiveState {
  pub fn new(batch_sz: usize, minibatch_sz: usize, momentum: Option<f32>, grad_sz: usize, init_step: f32, test_iters: usize, epoch_iters: usize, sched: AdaptiveStepSizeSchedule) -> SgdAdaptiveState {
    let mut param_sav = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      param_sav.push(0.0);
    }
    let mut grad = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      grad.push(0.0);
    }
    let mut grad_acc = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      grad_acc.push(0.0);
    }
    SgdAdaptiveState{
      batch_sz:     batch_sz,
      minibatch_sz: minibatch_sz,
      momentum:     momentum,
      grad_sz:      grad_sz,
      init_step:    init_step,
      test_iters:   test_iters,
      epoch_iters:  epoch_iters,
      sched:        sched,
      prev_step:    1.0,
      param_sav:    param_sav,
      grad:         grad,
      grad_acc:     grad_acc,
    }
  }

  pub fn search<S, R, Op>(&mut self, iter_counter: usize, cache: &mut Vec<S>, operator: &mut Op, frozen_param_sav: &[f32], frozen_grad_acc: &[f32], samples: &mut Iterator<Item=S>) -> f32
  where Op: DiffOperatorInput<f32, S, Rng=R>, S: SampleWeight, R: Rng {
    let num_batches = (self.minibatch_sz + self.batch_sz - 1) / self.batch_sz;
    cache.clear();
    for mut sample in samples.take(self.minibatch_sz * self.test_iters) {
      sample.mix_weight(1.0 / self.minibatch_sz as f32);
      cache.push(sample);
    }

    self.param_sav.copy_from_slice(frozen_param_sav);
    self.grad_acc.copy_from_slice(frozen_grad_acc);

    let mut rhs = 0.0;
    if let Some(mu) = self.momentum {
      operator.update_param(mu, 1.0, &mut self.grad_acc, 0);
    }
    operator.save_rng_state();
    for i in 0 .. self.test_iters {
      operator.reset_loss();
      for batch in 0 .. num_batches {
        let batch_start = batch * self.batch_sz + i * self.minibatch_sz;
        let batch_end = min((batch+1) * self.batch_sz, self.minibatch_sz) + i * self.minibatch_sz;
        operator.load_data(&cache[batch_start .. batch_end]);
        operator.forward(OpPhase::Learning);
      }
      rhs += operator.store_loss();
    }
    if let Some(_) = self.momentum {
      operator.load_param(&mut self.param_sav, 0);
    }

    // FIXME(20161004): do increasing search too to get largest step size that works.
    //let mut test_step = self.adapt_state.as_ref().unwrap().prev_step;
    let mut test_step = self.init_step;
    let mut test_descent = None;
    let mut test_prev_diverged = false;
    let mut t = 0;
    loop {
      match self.sched {
        AdaptiveStepSizeSchedule::Pow10 => {
          test_step = self.init_step * adaptive_pow10_step_size_factor(t);
        }
        AdaptiveStepSizeSchedule::Decimal => {
          test_step = self.init_step * adaptive_decimal_step_size_factor(t);
        }
      }

      let mut lhs = 0.0;
      self.param_sav.copy_from_slice(frozen_param_sav);
      self.grad_acc.copy_from_slice(frozen_grad_acc);
      operator.load_param(&mut self.param_sav, 0);
      operator.restore_rng_state();
      for i in 0 .. self.test_iters {
        operator.reset_loss();
        operator.reset_grad();
        if let Some(mu) = self.momentum {
          operator.update_param(mu, 1.0, &mut self.grad_acc, 0);
        }
        for batch in 0 .. num_batches {
          let batch_start = batch * self.batch_sz + i * self.minibatch_sz;
          let batch_end = min((batch+1) * self.batch_sz, self.minibatch_sz) + i * self.minibatch_sz;
          operator.load_data(&cache[batch_start .. batch_end]);
          operator.forward(OpPhase::Learning);
          operator.backward();
        }
        /*if let Some(lambda) = self.l2_reg {
          operator.apply_grad_reg(Regularization::L2(lambda));
        }*/
        if let Some(_) = self.momentum {
          operator.load_param(&mut self.param_sav, 0);
        }
        operator.store_grad(&mut self.grad, 0);
        if let Some(mu) = self.momentum {
          self.grad_acc.reshape_mut(self.grad_sz).vector_scale(mu);
        } else {
          self.grad_acc.reshape_mut(self.grad_sz).set_constant(0.0);
        }
        self.grad_acc.reshape_mut(self.grad_sz).vector_add(-test_step, self.grad.reshape(self.grad_sz));
        operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
        operator.update_nondiff_param(iter_counter + i);
        operator.store_param(&mut self.param_sav, 0);
        lhs += operator.store_loss();
      }

      let descent = rhs - lhs;
      t += 1;
      if descent <= 0.0 || descent.is_nan() {
        test_prev_diverged = true;
        continue;
      }
      test_prev_diverged = false;
      match test_descent {
        None => {
          test_descent = Some((test_step, descent));
        }
        Some((best_step, best_descent)) => {
          if descent > best_descent {
            test_descent = Some((test_step, descent));
          } else if !test_prev_diverged {
            test_step = best_step;
            break;
          }
        }
      }
    }
    self.prev_step = test_step;
    println!("DEBUG: auto adaptive step size: attempts: {} step: {:e} descent: {:e}", t, test_step, test_descent.unwrap().1);
    test_step
  }
}
#[derive(Clone, Copy, Debug)]
pub struct SgdConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub momentum:     Option<f32>,
  pub l2_reg:       Option<f32>,
}

pub struct SgdWorker<T, S, R, Op> where R: Rng, Op: DiffOperatorInput<T, S> {
  cfg:          SgdConfig,
  iter_counter: usize,
  adapt_state:  Option<SgdAdaptiveState>,
  operator:     Op,
  cache:        Vec<S>,
  param_sz:     usize,
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  grad_acc:     Vec<T>,
  stats_it:     usize,
  stats:        ClassOptStats,
  _marker:      PhantomData<R>,
}

impl<S, R, Op> SgdWorker<f32, S, R, Op> where R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
  pub fn new(cfg: SgdConfig, operator: Op) -> SgdWorker<f32, S, R, Op> {
    let grad_sz = operator.param_len();
    let adapt_state = if let StepSize::Adaptive{init_step, test_iters, epoch_iters, sched} = cfg.step_size {
      Some(SgdAdaptiveState::new(cfg.batch_sz, cfg.minibatch_sz, cfg.momentum, grad_sz, init_step, test_iters, epoch_iters, sched))
    } else {
      None
    };
    let cache = if let StepSize::Adaptive{test_iters, ..} = cfg.step_size {
      Vec::with_capacity(cfg.minibatch_sz * test_iters)
    } else {
      Vec::with_capacity(cfg.minibatch_sz)
    };
    let mut param_saved = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      param_saved.push(0.0);
    }
    let mut grad = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      grad.push(0.0);
    }
    let mut grad_acc = Vec::with_capacity(grad_sz);
    for _ in 0 .. grad_sz {
      grad_acc.push(0.0);
    }
    SgdWorker{
      cfg:          cfg,
      iter_counter: 0,
      adapt_state:  adapt_state,
      operator:     operator,
      cache:        cache,
      param_sz:     grad_sz,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      _marker:      PhantomData,
    }
  }
}

impl<S, R, Op> OptWorker<f32, S> for SgdWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
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

    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = self.iter_counter / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      StepSize::Adaptive{epoch_iters, ..} => {
        if self.iter_counter % epoch_iters == 0 {
          self.adapt_state.as_mut().unwrap().search(self.iter_counter, &mut self.cache, &mut self.operator, &self.param_saved, &self.grad_acc, samples)
        } else {
          self.adapt_state.as_ref().unwrap().prev_step
        }
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
    /*if let Some(lambda) = self.cfg.l2_reg {
      self.operator.apply_grad_reg(Regularization::L2(lambda));
    }*/
    if let Some(_) = self.cfg.momentum {
      self.operator.load_param(&mut self.param_saved, 0);
    }
    self.operator.store_grad(&mut self.grad, 0);

    if let Some(mu) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.param_sz).vector_scale(mu);
    } else {
      self.grad_acc.reshape_mut(self.param_sz).set_constant(0.0);
    }
    self.grad_acc.reshape_mut(self.param_sz).vector_add(-step_size, self.grad.reshape(self.param_sz));

    self.operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
    self.operator.update_nondiff_param(self.iter_counter);
    self.operator.store_param(&mut self.param_saved, 0);

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (self.operator.store_loss() - self.stats.avg_loss);

    self.iter_counter += 1;
    //self.prev_step = step_size;
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

impl<S, R, Op> OptStats<ClassOptStats> for SgdWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S> {
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
