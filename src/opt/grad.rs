use prelude::*;
use data::{SampleWeight};
use opt::{StepSize, ClassOptStats};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use densearray::{Reshape, ReshapeMut};
//use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cmp::{min};
use std::marker::{PhantomData};
//use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct GradientDescentOptConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    StepSize,
  pub momentum:     Option<f32>,
  pub l2_reg:       Option<f32>,
}

pub struct GradientDescentOptWorker<T, S, R, Op> where R: Rng, Op: DiffOperatorInput<T, S> {
  cfg:          GradientDescentOptConfig,
  iter_counter: usize,
  prev_step:    f32,
  operator:     Op,
  cache:        Vec<S>,
  minicache:    Vec<S>,
  param_sz:     usize,
  param_saved:  Vec<T>,
  grad:         Vec<T>,
  grad_acc:     Vec<T>,
  stats_it:     usize,
  stats:        ClassOptStats,
  _marker:      PhantomData<R>,
}

impl<S, R, Op> GradientDescentOptWorker<f32, S, R, Op> where R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
  pub fn new(cfg: GradientDescentOptConfig, operator: Op) -> GradientDescentOptWorker<f32, S, R, Op> {
    //let batch_sz = cfg.batch_sz;
    let param_sz = operator.param_len();
    let mut param_saved = Vec::with_capacity(param_sz);
    for _ in 0 .. param_sz {
      param_saved.push(0.0);
    }
    let mut grad = Vec::with_capacity(param_sz);
    for _ in 0 .. param_sz {
      grad.push(0.0);
    }
    let mut grad_acc = Vec::with_capacity(param_sz);
    for _ in 0 .. param_sz {
      grad_acc.push(0.0);
    }
    GradientDescentOptWorker{
      cfg:          cfg,
      iter_counter: 0,
      prev_step:    0.0,
      operator:     operator,
      cache:        Vec::with_capacity(cfg.batch_sz),
      minicache:    Vec::with_capacity(cfg.minibatch_sz),
      param_sz:     param_sz,
      param_saved:  param_saved,
      grad:         grad,
      grad_acc:     grad_acc,
      stats_it:     0,
      stats:        Default::default(),
      _marker:      PhantomData,
    }
  }
}

impl<S, R, Op> OptWorker<f32, S> for GradientDescentOptWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S, Rng=R> {
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

    let mut use_minicache = false;
    match self.cfg.step_size {
      StepSize::BacktrackingLineSearch{decay, c} => {
        use_minicache = true;
        self.minicache.clear();
        for mut sample in samples.take(self.cfg.minibatch_sz) {
          sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
          self.minicache.push(sample);
        }
      }
      _ => {}
    }

    self.operator.save_rng_state();
    self.operator.reset_loss();
    self.operator.reset_grad();
    if let Some(mu) = self.cfg.momentum {
      self.operator.update_param(-self.prev_step * mu, 1.0, &mut self.grad_acc, 0);
    }
    for batch in 0 .. num_batches {
      if use_minicache {
        let batch_offset = batch * self.cfg.batch_sz;
        let batch_size = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
        self.operator.load_data(&self.minicache[batch_offset .. batch_offset + batch_size]);
      } else {
        let actual_batch_sz = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
        self.cache.clear();
        for mut sample in samples.take(actual_batch_sz) {
          sample.mix_weight(1.0 / self.cfg.minibatch_sz as f32);
          self.cache.push(sample);
        }
        self.operator.load_data(&self.cache);
      }
      self.operator.forward(OpPhase::Learning);
      self.operator.backward();
    }
    if let Some(lambda) = self.cfg.l2_reg {
      self.operator.apply_grad_reg(Regularization::L2(lambda));
    }
    if let Some(_) = self.cfg.momentum {
      self.operator.load_param(&mut self.param_saved, 0);
    }
    self.operator.store_grad(&mut self.grad, 0);
    if let Some(mu) = self.cfg.momentum {
      self.grad_acc.reshape_mut(self.param_sz).vector_scale(mu);
      self.grad_acc.reshape_mut(self.param_sz).vector_add(1.0, self.grad.reshape(self.param_sz));
    } else {
      self.grad_acc.copy_from_slice(&self.grad);
    }

    let mut use_minicache = false;
    let step_size = match self.cfg.step_size {
      StepSize::Constant(alpha) => {
        alpha
      }
      StepSize::Decay{init_step, step_decay, decay_iters} => {
        let num_decays = self.iter_counter / decay_iters;
        init_step * step_decay.powi(num_decays as i32)
      }
      StepSize::BacktrackingLineSearch{decay, c} => {
        self.operator.save_rng_state();
        let loss = {
          self.operator.restore_rng_state();
          self.operator.reset_loss();
          for batch in 0 .. num_batches {
            let batch_offset = batch * self.cfg.batch_sz;
            let batch_size = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
            self.operator.load_data(&self.minicache[batch_offset .. batch_offset + batch_size]);
            self.operator.forward(OpPhase::Learning);
          }
          self.operator.store_loss()
        };
        let mut grad_norm = None;
        if c > 0.0 {
          grad_norm = Some(self.grad_acc.reshape(self.param_sz).l2_norm());
        }
        let mut t = 1.0;
        loop {
          self.operator.restore_rng_state();
          self.operator.reset_loss();
          self.operator.update_param(-t, 1.0, &mut self.grad_acc, 0);
          for batch in 0 .. num_batches {
            let batch_offset = batch * self.cfg.batch_sz;
            let batch_size = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
            self.operator.load_data(&self.minicache[batch_offset .. batch_offset + batch_size]);
            self.operator.forward(OpPhase::Learning);
          }
          let lhs = self.operator.store_loss();
          let rhs = if c > 0.0 {
            loss - c * t * grad_norm.unwrap() * grad_norm.unwrap()
          } else {
            loss
          };
          println!("DEBUG: backtrack: lhs: {:e} rhs: {:e} t: {:e}", lhs, rhs, t);
          self.operator.load_param(&mut self.param_saved, 0);
          if lhs > rhs {
            t *= decay;
          } else {
            break;
          }
        }
        self.operator.restore_rng_state();
        t
      }
    };

    self.operator.update_param(-step_size, 1.0, &mut self.grad_acc, 0);
    self.operator.store_param(&mut self.param_saved, 0);

    self.stats_it += 1;
    self.stats.sample_count += self.cfg.minibatch_sz;
    self.stats.avg_loss += 1.0 / (self.stats_it as f32) * (self.operator.store_loss() - self.stats.avg_loss);

    self.iter_counter += 1;
    self.prev_step = step_size;
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

impl<S, R, Op> OptStats<ClassOptStats> for GradientDescentOptWorker<f32, S, R, Op> where S: SampleWeight, R: Rng, Op: DiffOperatorInput<f32, S> {
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
