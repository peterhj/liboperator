use super::{DiffOperatorInput, CheckpointFormat, OpPhase};
use data::{SampleWeight};
use rw::{ReadBuffer, WriteBuffer};

use densearray::{Reshape, ReshapeMut};

use rand::{Rng};
use std::cmp::{min};
use std::path::{Path};

//pub mod adagrad;
pub mod adam;
//pub mod grad;
pub mod rmsprop;
pub mod sgd;

#[derive(Clone, Copy, Debug)]
pub enum StepSize {
  Constant(f32),
  Decay{init_step: f32, step_decay: f32, decay_iters: usize},
  Adaptive{init_step: f32, test_iters: usize, epoch_iters: usize, sched: AdaptiveStepSizeSchedule},
  BacktrackingLineSearch{decay: f32, c: f32},
  //WeakWolfeLineSearch,
}

#[derive(Clone, Copy, Debug)]
pub enum AdaptiveStepSizeSchedule {
  Pow10,
  Decimal,
}

pub fn adaptive_pow10_step_size_factor(t: usize) -> f32 {
  const F1: [f32; 2] = [0.3,    0.1];
  const F2: [f32; 2] = [0.03,   0.01];
  const F3: [f32; 2] = [0.003,  0.001];
  const F4: [f32; 2] = [0.0003, 0.0001];
  if t == 0 {
    1.0
  } else if t > 0 && t <= 2 {
    F1[t-1]
  } else if t > 2 && t <= 4 {
    F2[t-2-1]
  } else if t > 4 && t <= 6 {
    F3[t-4-1]
  } else if t > 6 && t <= 8 {
    F4[t-6-1]
  } else {
    unimplemented!();
  }
}

pub fn adaptive_decimal_step_size_factor(t: usize) -> f32 {
  const F1: [f32; 9] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
  const F2: [f32; 9] = [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01];
  const F3: [f32; 9] = [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001];
  if t == 0 {
    1.0
  } else if t > 0 && t <= 9 {
    F1[t-1]
  } else if t > 9 && t <= 18 {
    F2[t-9-1]
  } else if t > 18 && t <= 27 {
    F3[t-18-1]
  } else {
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
pub enum GradientMomentum {
  HeavyBall(f32),
  Nesterov(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum NesterovParamState {
  Orig,
  PlusMomentum,
}

#[derive(Clone, Copy, Debug)]
pub struct StepStats {
  pub elapsed:  Option<f64>,
}

pub trait OptWorker<T, S> {
  type Rng: Rng;

  //fn init_param(&mut self, rng: &mut Xorshiftplus128Rng);
  fn init_param(&mut self, rng: &mut Self::Rng);
  fn load_local_param(&mut self, param_reader: &mut ReadBuffer<T>);
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<T>);
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<T>);

  fn step(&mut self, samples: &mut Iterator<Item=S>) /* -> StepStats*/;
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
