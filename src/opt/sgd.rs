use super::super::{Operator, OpPhase};
use data::{SampleCastAs};
use opt::{OptWorker};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

use std::cmp::{min};
//use std::marker::{PhantomData};

//#[derive(Clone, Copy)]
pub struct SgdOptConfig {
  pub batch_size:       usize,
  pub minibatch_size:   usize,
  pub step_size:        f32,
  pub momentum:         Option<f32>,
}

pub struct SgdOptWorker<T, Op> where Op: Operator<T> {
  cfg:      SgdOptConfig,
  operator: Op,
  //cache:    Vec<<Op as Operator<T>>::Sample>,
  grad_acc: Vec<T>,
}

impl<Op> SgdOptWorker<f32, Op> where Op: Operator<f32> {
  pub fn new(cfg: SgdOptConfig, operator: Op) -> SgdOptWorker<f32, Op> {
    let batch_size = cfg.batch_size;
    let param_len = operator.param_len();
    let mut grad_acc = Vec::with_capacity(param_len);
    for _ in 0 .. param_len {
      grad_acc.push(0.0);
    }
    SgdOptWorker{
      cfg:      cfg,
      operator: operator,
      //cache:    Vec::with_capacity(batch_size),
      grad_acc: grad_acc,
    }
  }
}

impl<Op> OptWorker<f32, Op> for SgdOptWorker<f32, Op> where Op: Operator<f32> {
  fn init_param(&mut self) {
    self.operator.init_param();
  }

  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<f32>) {
  }

  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<f32>) {
  }

  //fn step(&mut self, samples: &mut Iterator<Item=<Op as Operator<f32>>::Sample>) {
  fn step<S>(&mut self, samples: &mut Iterator<Item=S>)
  where S: SampleCastAs<<Op as Operator<f32>>::Sample>
  {
    let mut cache = Vec::with_capacity(self.cfg.batch_size);
    let num_batches = (self.cfg.minibatch_size + self.cfg.batch_size - 1) / self.cfg.batch_size;
    for batch in 0 .. num_batches {
      let actual_batch_size = min((batch+1) * self.cfg.batch_size, self.cfg.minibatch_size) - batch * self.cfg.batch_size;
      cache.clear();
      for sample in samples.take(actual_batch_size) {
        cache.push(sample);
      }
      self.operator.load_data(&cache);
      self.operator.forward(OpPhase::Learning);
      self.operator.backward();
    }
    self.operator.accumulate_grad(self.cfg.step_size, 0.0, &mut self.grad_acc);
    self.operator.update_param(1.0, 1.0, &mut self.grad_acc);
  }

  //fn eval(&mut self, samples: &mut Iterator<Item=<Op as Operator<f32>>::Sample>) {
  fn eval<S>(&mut self, samples: &mut Iterator<Item=S>)
  where S: SampleCastAs<<Op as Operator<f32>>::Sample>
  {
  }
}
