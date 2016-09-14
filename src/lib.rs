extern crate rand;
extern crate rng;

use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};

pub mod data;
pub mod opt;
pub mod rw;

#[derive(Clone, Copy, Debug)]
pub enum OpPhase {
  Inference,
  Learning,
}

pub trait Operator<T, S>: InternalOperator<T> {
  fn load_data(&mut self, samples: &[S]);
}

pub trait InternalOperator<T> {
  type Output: Clone;

  fn output(&self, arm: usize) -> Self::Output;
  fn param_len(&self) -> usize { 0 }
  //fn grad_len(&self) -> usize;

  fn save_rng_state(&mut self) {}
  fn restore_rng_state(&mut self) {}

  fn init_state(&mut self) {}

  fn init_param(&mut self, _rng: &mut Xorshiftplus128Rng) {}
  fn load_param(&mut self, _param_reader: &mut ReadBuffer<T>) -> usize { 0 }
  fn store_param(&mut self, _param_writer: &mut WriteBuffer<T>) -> usize { 0 }
  fn update_param(&mut self, _alpha: f32, _beta: f32, _grad_reader: &mut ReadAccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn reset_grad(&mut self) {}
  fn load_grad(&mut self, _grad_reader: &mut ReadBuffer<T>) -> usize { 0 }
  fn store_grad(&mut self, _grad_writer: &mut WriteBuffer<T>) -> usize { 0 }
  fn apply_l2_reg(&mut self, _lambda: f32) {}
  fn accumulate_grad(&mut self, _alpha: f32, _beta: f32, _grad_accum: &mut AccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn forward(&mut self, phase: OpPhase);
  fn backward(&mut self);
  fn r_forward(&mut self) { unimplemented!(); }
  fn r_backward(&mut self) { unimplemented!(); }
}
