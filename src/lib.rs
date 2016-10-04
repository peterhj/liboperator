extern crate densearray;

extern crate rand;

use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rand::{Rng};
use std::io::{Read, Write};
use std::ops::{Deref};

pub mod data;
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

#[derive(Clone, Copy, Debug)]
pub enum OpPhase {
  Inference,
  Learning,
}

#[derive(Clone, Copy, Debug)]
pub enum Regularization {
  L2(f32),
}

pub trait DiffOperator<T> {
  type Output;
  type Rng: Rng;

  fn _output(&self, arm: usize) -> Self::Output;

  #[deprecated] fn param_len(&self) -> usize { 0 }
  fn diff_param_sz(&self) -> usize { 0 }
  fn nondiff_param_sz(&self) -> usize { 0 }
  fn total_param_sz(&self) -> usize {
    self.diff_param_sz() + self.nondiff_param_sz()
  }

  fn save_rng_state(&mut self) {}
  fn restore_rng_state(&mut self) {}
  fn store_rng_state(&mut self, _rng_state: &mut Write) {}
  fn load_rng_state(&mut self, _rng_state: &mut Read) {}

  //fn init_state(&mut self) {}

  fn init_param(&mut self, _rng: &mut Self::Rng) {}
  fn reset_nondiff_param(&mut self) {}
  fn store_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn store_nondiff_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_nondiff_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn update_param(&mut self, _alpha: f32, _beta: f32, _grad_reader: &mut ReadAccumulateBuffer<T>, _offset: usize) -> usize { 0 }
  fn update_nondiff_param(&mut self, iter: usize) {}

  fn reset_grad(&mut self) {}
  fn store_grad(&mut self, _grad_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  //fn load_grad(&mut self, _grad_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn accumulate_grad(&mut self, _alpha: f32, _beta: f32, _grad_accum: &mut AccumulateBuffer<T>, _offset: usize) -> usize { 0 }
  fn grad_step(&mut self, _alpha: f32, _beta: f32) {}

  fn reset_loss(&mut self) {}
  fn store_loss(&mut self) -> f32 { 0.0 }
  fn add_loss(&mut self, _extra_loss: f32) { unimplemented!(); }

  fn apply_grad_reg(&mut self, _reg: Regularization) {}
  //fn apply_reg(&mut self, _reg: Regularization) {}

  fn forward(&mut self, phase: OpPhase);
  fn fwd_reg(&mut self, _reg: Regularization) {}
  fn backward(&mut self);
  fn bwd_reg(&mut self, _reg: Regularization) {}
  fn r_forward(&mut self) { unimplemented!(); }
  fn r_backward(&mut self) { unimplemented!(); }
}

pub trait DiffOperatorInput<T, S>: DiffOperator<T> {
  fn load_data(&mut self, samples: &[S]);
}

pub trait DiffOperatorOutput<T, U>: DiffOperator<T> {
  fn get_output(&mut self) -> U;
}

pub trait DiffOperatorIo<T, S, U>: DiffOperatorInput<T, S> + DiffOperatorOutput<T, U> {
}

pub trait CheckpointFormat {
}

pub trait DiffOperatorCheckpoint<Format> where Format: CheckpointFormat {
  fn decode(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn encode(&mut self, writer: &mut Write) -> Result<(), ()>;
}
