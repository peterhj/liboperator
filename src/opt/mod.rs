use super::{Operator};
use data::{SampleCastAs};
use rw::{ReadBuffer, WriteBuffer};

//use rand::{Rng};

pub mod sgd;

pub trait OptWorker<T, S, Op> where Op: Operator<T, S> { //, S: SampleCastAs<<Op as Operator<T, S>>::Sample> {
  fn init_param(&mut self);
  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<T>);
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<T>);
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<T>);

  /*fn step<S>(&mut self, samples: &mut Iterator<Item=S>)
  where S: SampleCastAs<<Op as Operator<T>>::Sample>;
  fn eval<S>(&mut self, samples: &mut Iterator<Item=S>)
  where S: SampleCastAs<<Op as Operator<T>>::Sample>;*/
  fn step(&mut self, samples: &mut Iterator<Item=S>);
  fn eval(&mut self, samples: &mut Iterator<Item=S>);
}

pub trait OptStats<St> {
  fn reset_stats(&mut self, stats: &mut St);
  fn get_stats(&mut self, stats: &mut St);
}

pub struct ClassStats {
  pub sample_count:     usize,
  pub correct_count:    usize,
  pub accuracy:         f32,
  pub avg_loss:         f32,
}

pub struct RegressStats {
  pub avg_loss:         f32,
}
