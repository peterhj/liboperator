extern crate densearray;
extern crate rng;

extern crate rand;

use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};

pub mod data;
pub mod memory;
pub mod opt;
pub mod prelude;
pub mod rw;

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

pub trait Operator<T, S>: DiffOperator<T> {
  fn load_data(&mut self, samples: &[S]);
  fn store_loss(&mut self) -> f32 { unimplemented!(); }
}

pub trait DiffOperator<T> {
  type Output;

  fn output(&self, arm: usize) -> Self::Output;
  fn param_len(&self) -> usize { 0 }
  //fn grad_len(&self) -> usize;

  fn save_rng_state(&mut self) {}
  fn restore_rng_state(&mut self) {}

  //fn init_state(&mut self) {}

  fn init_param(&mut self, _rng: &mut Xorshiftplus128Rng) {}
  fn load_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn store_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn update_param(&mut self, _alpha: f32, _beta: f32, _grad_reader: &mut ReadAccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn reset_loss(&mut self) {}

  fn reset_grad(&mut self) {}
  fn load_grad(&mut self, _grad_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn store_grad(&mut self, _grad_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn apply_grad_reg(&mut self, _reg: Regularization) {}
  fn accumulate_grad(&mut self, _alpha: f32, _beta: f32, _grad_accum: &mut AccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn forward(&mut self, phase: OpPhase);
  fn backward(&mut self);
  fn r_forward(&mut self) { unimplemented!(); }
  fn r_backward(&mut self) { unimplemented!(); }
}
