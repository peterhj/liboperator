use super::super::{Operator, OpPhase};
use opt::{OptWorker};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cmp::{min};
use std::marker::{PhantomData};

#[derive(Clone, Copy)]
pub struct SgdOptConfig {
  pub batch_sz:     usize,
  pub minibatch_sz: usize,
  pub step_size:    f32,
  pub momentum:     Option<f32>,
}

pub struct SgdOptWorker<T, S, Op> where Op: Operator<T, S> {
  cfg:      SgdOptConfig,
  operator: Op,
  cache:    Vec<S>,
  grad_acc: Vec<T>,
  //_marker:  PhantomData<S>,
}

impl<S, Op> SgdOptWorker<f32, S, Op> where Op: Operator<f32, S> {
  pub fn new(cfg: SgdOptConfig, operator: Op) -> SgdOptWorker<f32, S, Op> {
    let batch_sz = cfg.batch_sz;
    let param_len = operator.param_len();
    let mut grad_acc = Vec::with_capacity(param_len);
    for _ in 0 .. param_len {
      grad_acc.push(0.0);
    }
    SgdOptWorker{
      cfg:      cfg,
      operator: operator,
      cache:    Vec::with_capacity(batch_sz),
      grad_acc: grad_acc,
      //_marker:  PhantomData,
    }
  }
}

impl<S, Op> OptWorker<f32, S, Op> for SgdOptWorker<f32, S, Op> where Op: Operator<f32, S> {
  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    self.operator.init_param(rng);
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn step(&mut self, samples: &mut Iterator<Item=S>) {
    let mut cache = Vec::with_capacity(self.cfg.batch_sz);
    let num_batches = (self.cfg.minibatch_sz + self.cfg.batch_sz - 1) / self.cfg.batch_sz;
    for batch in 0 .. num_batches {
      let actual_batch_sz = min((batch+1) * self.cfg.batch_sz, self.cfg.minibatch_sz) - batch * self.cfg.batch_sz;
      cache.clear();
      for sample in samples.take(actual_batch_sz) {
        cache.push(sample);
      }
      self.operator.load_data(&cache);
      self.operator.forward(OpPhase::Learning);
      self.operator.backward();
    }
    self.operator.accumulate_grad(self.cfg.step_size, 0.0, &mut self.grad_acc, 0);
    self.operator.update_param(1.0, 1.0, &mut self.grad_acc, 0);
  }

  fn eval(&mut self, samples: &mut Iterator<Item=S>) {
  }
}
