use super::{CheckpointFormat};
use rw::{ReadBuffer, WriteBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::path::{Path};

pub mod sgd;

pub trait OptWorker<T, S> {
  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng);
  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<T>);
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<T>);
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<T>);

  fn step(&mut self, samples: &mut Iterator<Item=S>);
  fn eval(&mut self, epoch_size: usize, samples: &mut Iterator<Item=S>);
}

pub trait OptCheckpoint<Format> where Format: CheckpointFormat {
  fn save_checkpoint(&mut self, path: &Path) -> Result<(), ()>;
  fn restore_checkpoint(&mut self, path: &Path) -> Result<(), ()>;
}

pub trait OptStats<Stats> {
  fn reset_opt_stats(&mut self);
  fn get_opt_stats(&self) -> &Stats;
}

#[derive(Clone, Default, Debug)]
pub struct ClassOptStats {
  pub sample_count:     usize,
  pub correct_count:    usize,
  pub avg_loss:         f32,
}

impl ClassOptStats {
  pub fn accuracy(&self) -> f32 {
    self.correct_count as f32 / self.sample_count as f32
  }
}

#[derive(Clone, Default, Debug)]
pub struct RegressOptStats {
  pub avg_loss:         f32,
}
