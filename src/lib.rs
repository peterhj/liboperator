#![feature(conservative_impl_trait)]

extern crate densearray;
extern crate rng;
extern crate sharedmem;

extern crate rand;

//use io::{IoBuffer};
use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cell::{Cell};
use std::io::{Read, Write};
use std::ops::{Deref};

pub mod data;
pub mod io;
pub mod opt;
pub mod prelude;
pub mod rw;
pub mod timing;

#[derive(Clone, Copy)]
pub enum OpCapability {
  Forward,
  Backward,
  RForward,
  RBackward,
}

impl OpCapability {
  pub fn enable_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      _ => true,
    }
  }

  pub fn enable_r_forward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      _ => true,
    }
  }

  pub fn enable_r_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      OpCapability::RForward => false,
      _ => true,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpPhase {
  Inference,
  Learning,
}

#[derive(Clone, Copy, Debug)]
pub enum Regularization {
  L2(f32),
}

pub trait Operator {
  fn _next(&self) -> u64;
  fn _epoch(&self) -> u64;
}

#[derive(Clone, Default)]
pub struct OperatorNode {
  pub curr_epoch:   Cell<u64>,
  pub curr_count:   Cell<u64>,
}

impl Operator for OperatorNode {
  fn _next(&self) -> u64 {
    self.curr_epoch.set(self.curr_epoch.get() + 1);
    self.curr_count.set(0);
    self.curr_epoch.get()
  }

  fn _epoch(&self) -> u64 {
    self.curr_epoch.get()
  }
}

impl OperatorNode {
  pub fn step(&self, next_epoch: u64) {
    assert!(next_epoch >= self.curr_epoch.get());
    if next_epoch > self.curr_epoch.get() {
      self.curr_epoch.set(next_epoch);
      self.curr_count.set(0);
    }
    self.curr_count.set(self.curr_count.get() + 1);
  }

  pub fn count(&self) -> u64 {
    self.curr_count.get()
  }

  pub fn limit(&self, max_count: u64) -> bool {
    self.curr_count.get() <= max_count
  }
}

pub trait OperatorOutput {
}

pub trait NewDiffOperator<S>: Operator {
  //type Output;
  type IoBuf: ?Sized;

  //fn _op_output(&self, arm: usize) -> impl OperatorOutput;
  //fn _output(&self, arm: usize) -> Self::Output;
  fn _traverse_fwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>));
  fn _traverse_bwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>));
  //fn _traverse_fwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, Output=Self::Output, IoBuf=Self::IoBuf>));
  //fn _traverse_bwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, Output=Self::Output, IoBuf=Self::IoBuf>));

  fn _diff_param_sz(&self) -> usize { 0 }
  fn _nondiff_param_sz(&self) -> usize { 0 }

  fn _load_diff_param(&mut self, offset: usize, param_reader: &mut Self::IoBuf) -> usize { 0 }
  fn _load_nondiff_param(&mut self, offset: usize, param_reader: &mut Self::IoBuf) -> usize { 0 }
  fn _store_diff_param(&mut self, offset: usize, param_writer: &mut Self::IoBuf) -> usize { 0 }
  fn _store_nondiff_param(&mut self, offset: usize, param_writer: &mut Self::IoBuf) -> usize { 0 }
  fn _store_grad(&mut self, offset: usize, grad_writer: &mut Self::IoBuf) -> usize { 0 }

  fn _save_rng_state(&mut self) {}
  fn _restore_rng_state(&mut self) {}

  fn _next_iteration(&mut self) {}
  fn _load_batch(&mut self, _samples: &[S]) {}
  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {}
  fn _update_nondiff_param(&mut self, _iter_nr: usize) {}
  fn _reset_grad(&mut self) {}

  fn _forward(&mut self, phase: OpPhase);
  fn _backward(&mut self);
}

pub trait DiffLoss<S>: NewDiffOperator<S> {
  fn reset_loss(&mut self);
  fn store_loss(&mut self) -> f32;
  fn _store_accuracy(&mut self) -> usize { 0 }

  fn diff_param_sz(&mut self) -> usize {
    let epoch = self._next();
    let mut grad_sz = 0;
    self._traverse_bwd(epoch, &mut |op| {
      grad_sz += op._diff_param_sz();
    });
    grad_sz
  }

  fn nondiff_param_sz(&mut self) -> usize {
    let epoch = self._next();
    let mut nondiff_sz = 0;
    self._traverse_bwd(epoch, &mut |op| {
      nondiff_sz += op._nondiff_param_sz();
    });
    nondiff_sz
  }

  fn load_diff_param(&mut self, param_reader: &mut Self::IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_diff_param(offset, param_reader);
    });
    offset
  }

  fn load_nondiff_param(&mut self, param_reader: &mut Self::IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_nondiff_param(offset, param_reader);
    });
    offset
  }

  fn store_diff_param(&mut self, param_writer: &mut Self::IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_diff_param(offset, param_writer);
    });
    offset
  }

  fn store_nondiff_param(&mut self, param_writer: &mut Self::IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_nondiff_param(offset, param_writer);
    });
    offset
  }

  fn store_grad(&mut self, grad_writer: &mut Self::IoBuf) -> usize {
    //grad_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_grad(offset, grad_writer);
    });
    offset
  }

  fn next_iteration(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._next_iteration());
  }

  fn load_batch(&mut self, samples: &[S]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._load_batch(samples));
  }

  fn save_rng_state(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._save_rng_state());
  }

  fn restore_rng_state(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._restore_rng_state());
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._init_param(rng));
  }

  fn update_nondiff_param(&mut self, iter_nr: usize) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._update_nondiff_param(iter_nr));
  }

  fn reset_grad(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._reset_grad());
  }

  fn forward(&mut self, phase: OpPhase) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._forward(phase));
  }

  fn backward(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._backward());
  }
}

pub trait DiffOperator<T> {
  type Output;
  type Rng: Rng;

  fn _output(&self, arm: usize) -> Self::Output;

  fn diff_param_sz(&self) -> usize { 0 }
  fn nondiff_param_sz(&self) -> usize { 0 }
  fn total_param_sz(&self) -> usize {
    self.diff_param_sz() + self.nondiff_param_sz()
  }

  fn save_rng_state(&mut self) {}
  fn restore_rng_state(&mut self) {}
  //fn store_rng_state(&mut self, _rng_state: &mut Write) {}
  //fn load_rng_state(&mut self, _rng_state: &mut Read) {}

  //fn reset_state(&mut self) {}

  fn init_param(&mut self, _rng: &mut Self::Rng) {}
  //fn reset_nondiff_param(&mut self) {}
  fn reset_grad(&mut self) {}

  fn store_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn store_nondiff_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_nondiff_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn update_param(&mut self, _alpha: f32, _beta: f32, _grad_reader: &mut ReadAccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn update_nondiff_param(&mut self, _iter: usize) {}

  fn store_grad(&mut self, _grad_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn accumulate_grad(&mut self, _alpha: f32, _beta: f32, _grad_accum: &mut AccumulateBuffer<T>, _offset: usize) -> usize { 0 }
  //fn grad_step(&mut self, _alpha: f32, _beta: f32) {}

  fn reset_loss(&mut self) {}
  fn store_loss(&mut self) -> f32 { 0.0 }
  fn _store_accuracy(&mut self) -> usize { 0 }

  fn apply_grad_reg(&mut self, _reg: Regularization) {}

  fn forward(&mut self, phase: OpPhase);
  //fn fwd_reg(&mut self, _reg: Regularization) {}
  fn backward(&mut self);
  //fn bwd_reg(&mut self, _reg: Regularization) {}
  //fn r_forward(&mut self) { unimplemented!(); }
  //fn r_backward(&mut self) { unimplemented!(); }
}

pub trait DiffOperatorIo<T, Buf>/*: DiffOperator*/ where T: Copy {
  fn store_diff_param(&mut self, _param_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn store_nondiff_param(&mut self, _param_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn load_diff_param(&mut self, _param_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn load_nondiff_param(&mut self, _param_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn update_diff_param(&mut self, _alpha: T, _beta: T, _delta_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn store_grad(&mut self, _grad_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn accumulate_grad(&mut self, _alpha: T, _beta: T, _grad_writer: &mut Buf, _offset: usize) -> usize { 0 }
}

pub trait DiffOperatorInput<T, S>: DiffOperator<T> {
  fn as_op(&self) -> &DiffOperator<T, Output=Self::Output, Rng=Self::Rng> { unimplemented!(); }
  fn load_data(&mut self, samples: &[S]);
}

pub trait DiffOperatorOutput<T, U>: DiffOperator<T> {
  fn get_output(&mut self) -> U;
}

pub trait CheckpointFormat {
}

pub trait DiffOperatorCheckpoint<Format> where Format: CheckpointFormat {
  fn decode(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn encode(&mut self, writer: &mut Write) -> Result<(), ()>;
}
