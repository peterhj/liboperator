use super::{DiffOperatorInput, CheckpointFormat, OpPhase};
use rw::{ReadBuffer, WriteBuffer};

use densearray::{Reshape, ReshapeMut};

use rand::{Rng};
use std::cmp::{max, min};
use std::fs::{File, create_dir_all};
use std::io::{Write};
use std::path::{Path, PathBuf};

//pub mod adagrad;
pub mod adam;
pub mod adam_new;
pub mod rmsprop;
pub mod sgd;
pub mod sgd_new;
pub mod shared_sgd;

#[derive(Clone, Debug)]
pub struct CheckpointConfig {
  pub prefix:   PathBuf,
  pub trace:    bool,
}

impl CheckpointConfig {
  pub fn build_state(&self) -> CheckpointState {
    create_dir_all(&self.prefix).ok();
    let mut lo_idx = Some(0);
    let mut hi_idx = None;
    let mut idx = 0;
    while lo_idx != hi_idx {
      match (lo_idx, hi_idx) {
        (Some(lo), None) => {
          idx = lo;
        }
        (Some(lo), Some(hi)) => {
          idx = (lo + hi) / 2;
        }
        _ => unreachable!(),
      }
      let mut config_path = PathBuf::from(&self.prefix);
      let mut config_filename = format!("config.{}", idx);
      config_path.push(&config_filename);
      if config_path.exists() {
        match (lo_idx, hi_idx) {
          (Some(lo), None) => {
            lo_idx = Some(lo + max(1, lo));
          }
          (Some(lo), Some(hi)) => {
            lo_idx = Some(min(hi, max(lo + 1, idx)));
          }
          _ => unreachable!(),
        }
      } else {
        match (lo_idx, hi_idx) {
          (Some(_), None) => {
            hi_idx = Some(idx);
          }
          (Some(_), Some(_)) => {
            hi_idx = Some(idx);
          }
          _ => unreachable!(),
        }
      }
    }
    idx = hi_idx.unwrap();
    let mut cfg_f = None;
    let mut trace_f = None;
    let mut valid_f = None;
    loop {
      let mut config_path = PathBuf::from(&self.prefix);
      let mut config_filename = format!("config.{}", idx);
      config_path.push(&config_filename);
      match File::create(&config_path) {
        Err(_) => {
          idx += 1;
          continue;
        }
        Ok(f) => {
          cfg_f = Some(f);
          break;
        }
      }
    }
    if self.trace {
      let mut train_path = PathBuf::from(&self.prefix);
      let mut train_filename = format!("train.log.{}", idx);
      train_path.push(&train_filename);
      assert!(!train_path.exists());
      let mut f = File::create(&train_path).unwrap();
      writeln!(&mut f, "iter,loss,other,elapsed,clock").unwrap();
      trace_f = Some(f);

      let mut valid_path = PathBuf::from(&self.prefix);
      let mut valid_filename = format!("valid.log.{}", idx);
      valid_path.push(&valid_filename);
      assert!(!valid_path.exists());
      let mut f = File::create(&valid_path).unwrap();
      writeln!(&mut f, "iter,val_loss,val_other,clock").unwrap();
      valid_f = Some(f);
    }
    CheckpointState{
      config_file:  cfg_f,
      train_file:   trace_f,
      valid_file:   valid_f,
    }
  }

  pub fn write_iteration_trace(&self, file: &mut File, iter_nr: usize, loss: f32, valid_loss: Option<f32>, other: f32, valid_other: Option<f32>, elapsed_s: f32) {
    if valid_loss.is_some() && valid_other.is_some() {
      writeln!(file, "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6}", iter_nr, loss, valid_loss.unwrap(), other, valid_other.unwrap(), elapsed_s).unwrap();
    } else if valid_loss.is_some() {
      writeln!(file, "{},{:.6e},{:.6e},{:.6e},,{:.6}", iter_nr, loss, valid_loss.unwrap(), other, elapsed_s).unwrap();
    } else if valid_other.is_some() {
      writeln!(file, "{},{:.6e},,{:.6e},{:.6e},{:.6}", iter_nr, loss, other, valid_other.unwrap(), elapsed_s).unwrap();
    } else {
      writeln!(file, "{},{:.6e},,{:.6e},,{:.6}", iter_nr, loss, other, elapsed_s).unwrap();
    }
  }
}

#[derive(Default)]
pub struct CheckpointState {
  pub config_file:  Option<File>,
  pub train_file:   Option<File>,
  pub valid_file:   Option<File>,
}

pub trait TraceRecord {
  fn loss(&self) -> f32;
  fn other(&self) -> f32;
}

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

pub trait OptUpdateStats<Stats> {
  fn update_stats(&mut self, stats: &mut Stats);
}

#[derive(Clone, Default, Debug)]
pub struct ClassOptStats {
  pub iter_count:       usize,
  pub sample_count:     usize,
  pub correct_count:    usize,
  pub avg_loss:         f32,
}

impl ClassOptStats {
  pub fn reset(&mut self) {
    self.iter_count = 0;
    self.sample_count = 0;
    self.correct_count = 0;
    self.avg_loss = 0.0;
  }

  pub fn accuracy(&self) -> f32 {
    self.correct_count as f32 / self.sample_count as f32
  }
}

#[derive(Clone, Default, Debug)]
pub struct RegressOptStats {
  pub avg_loss:         f32,
}
