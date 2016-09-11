#![feature(conservative_impl_trait)]

extern crate array;
extern crate rand;

use data::{SampleCastAs};
use rw::{ReadBuffer, WriteBuffer, AccumulateBuffer};

pub mod data;
pub mod opt;
pub mod rw;

pub enum OpPhase {
  Inference,
  Learning,
}

pub trait Operator<T>: InternalOperator<T> {
  type Sample;

  fn load_data<S>(&mut self, samples: &[S]) where S: SampleCastAs<Self::Sample>;
}

pub trait ExternalOperator<T, S>: InternalOperator<T> where S: SampleCastAs<Self::Sample> {
  type Sample;

  fn load_data(&mut self, samples: &[S]);
}

pub trait InternalOperator<T> {
  type Output: Clone;

  fn output(&self, arm: usize) -> Self::Output;
  fn param_len(&self) -> usize;
  //fn grad_len(&self) -> usize;

  fn save_rng_state(&mut self);
  fn restore_rng_state(&mut self);

  fn init_state(&mut self);

  fn init_param(&mut self);
  fn load_param(&mut self, param_reader: &mut ReadBuffer<T>);
  fn store_param(&mut self, param_writer: &mut WriteBuffer<T>);
  fn update_param(&mut self, alpha: f32, beta: f32, grad_reader: &mut ReadBuffer<T>);

  fn reset_grad(&mut self);
  fn load_grad(&mut self, grad_reader: &mut ReadBuffer<T>);
  fn store_grad(&mut self, grad_writer: &mut WriteBuffer<T>);
  fn accumulate_grad(&mut self, step_size: f32, mu: f32, grad_accum: &mut AccumulateBuffer<T>);

  fn forward(&mut self, phase: OpPhase);
  fn backward(&mut self);
  fn r_forward(&mut self) { unimplemented!(); }
  fn r_backward(&mut self) { unimplemented!(); }
}
