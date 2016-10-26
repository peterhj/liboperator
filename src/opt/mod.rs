use prelude::*;
use super::{CheckpointFormat};
use rw::{ReadBuffer, WriteBuffer};

use csv::{Writer as CsvWriter};
use densearray::{Reshape, ReshapeMut};

use rand::{Rng, thread_rng};
use rustc_serialize::{Encodable};
use std::cmp::{max, min};
use std::fmt::{Debug};
use std::fs::{File, create_dir_all};
use std::io::{Write};
use std::path::{Path, PathBuf};
use std::thread::{sleep};
use std::time::{Duration, Instant};

//pub mod adagrad;
pub mod adagrad_new;
//pub mod adam;
pub mod adam_new;
//pub mod rmsprop;
//pub mod sgd;
pub mod sgd_new;
//pub mod shared_sgd;
pub mod shared_sgd_new;

#[derive(Clone, Debug)]
pub struct CheckpointConfig {
  pub prefix:   PathBuf,
  pub trace:    bool,
}

pub struct CheckpointState {
  pub cfg:          CheckpointConfig,
  pub config_file:  Option<File>,
  pub train_file:   Option<CsvWriter<File>>,
  pub valid_file:   Option<CsvWriter<File>>,
  start_time:   Instant,
  lap_time:     Instant,
  elapsed_s:    f64,
}

impl CheckpointState {
  pub fn new(cfg: CheckpointConfig) -> CheckpointState {
    Self::with_fields(cfg, "iter,loss,accuracy,elapsed")
  }

  pub fn with_fields(cfg: CheckpointConfig, fields: &str) -> CheckpointState {
    let delay_s = thread_rng().gen_range(0, 2);
    let delay_ns = thread_rng().gen_range(100_000_000, 500_000_000);
    sleep(Duration::new(delay_s, delay_ns));
    create_dir_all(&cfg.prefix).ok();
    loop {
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
        let mut config_path = PathBuf::from(&cfg.prefix);
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
              lo_idx = Some(idx / 2);
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

      let delay_s = thread_rng().gen_range(0, 2);
      let delay_ns = thread_rng().gen_range(100_000_000, 500_000_000);
      sleep(Duration::new(delay_s, delay_ns));
      let mut cfg_f = None;
      let mut trace_f = None;
      let mut valid_f = None;
      let mut config_path = PathBuf::from(&cfg.prefix);
      let mut config_filename = format!("config.{}", idx);
      config_path.push(&config_filename);
      if config_path.exists() {
        continue;
      }
      match File::create(&config_path) {
        Err(_) => {
          continue;
        }
        Ok(f) => {
          cfg_f = Some(f);
        }
      }
      if cfg.trace {
        let mut train_path = PathBuf::from(&cfg.prefix);
        let mut train_filename = format!("train.log.{}", idx);
        train_path.push(&train_filename);
        assert!(!train_path.exists());
        let mut f = File::create(&train_path).unwrap();
        writeln!(&mut f, "{}", fields).unwrap();
        trace_f = Some(f);

        let mut valid_path = PathBuf::from(&cfg.prefix);
        let mut valid_filename = format!("valid.log.{}", idx);
        valid_path.push(&valid_filename);
        assert!(!valid_path.exists());
        let mut f = File::create(&valid_path).unwrap();
        writeln!(&mut f, "{}", fields).unwrap();
        valid_f = Some(f);
      }
      let init_time = Instant::now();
      return CheckpointState{
        cfg:          cfg,
        config_file:  cfg_f,
        train_file:   trace_f.map(|f| CsvWriter::from_writer(f)),
        valid_file:   valid_f.map(|f| CsvWriter::from_writer(f)),
        start_time:   init_time,
        lap_time:     init_time,
        elapsed_s:    0.0,
      };
    }
  }

  pub fn start_timing(&mut self) {
    self.start_time = Instant::now();
  }

  pub fn stop_timing(&mut self) {
    self.lap_time = Instant::now();
    let duration = self.lap_time - self.start_time;
    self.elapsed_s = duration.as_secs() as f64 + 1.0e-9 * duration.subsec_nanos() as f64;
  }

  pub fn elapsed(&self) -> f64 {
    self.elapsed_s
  }

  pub fn append_config_info<U>(&mut self, info: &U) where U: Debug {
    if let Some(ref mut config_file) = self.config_file {
      writeln!(config_file, "{:?}", info).unwrap();
      config_file.flush().unwrap();
    }
  }

  pub fn append_class_stats_train(&mut self, stats: &ClassLossStats) {
    let elapsed = self.elapsed_s;
    self.append_train_info(&stats.to_record(elapsed));
  }

  pub fn append_train_info<Rec>(&mut self, rec: &Rec) where Rec: Encodable {
    if let Some(ref mut train_file) = self.train_file {
      train_file.encode(rec).unwrap();
      train_file.flush().unwrap();
    }
  }

  pub fn append_class_stats_valid(&mut self, stats: &ClassLossStats) {
    let elapsed = self.elapsed_s;
    self.append_valid_info(&stats.to_record(elapsed));
  }

  pub fn append_valid_info<Rec>(&mut self, rec: &Rec) where Rec: Encodable {
    if let Some(ref mut valid_file) = self.valid_file {
      valid_file.encode(rec).unwrap();
      valid_file.flush().unwrap();
    }
  }
}

pub trait TraceRecord {
  fn loss(&self) -> f32;
  fn other(&self) -> f32;
}

#[derive(Clone, Debug)]
pub enum StepSize {
  Constant(f32),
  Custom{init_step: f32, steps: Vec<(usize, f32)>},
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
  /*fn load_local_param(&mut self, param_reader: &mut ReadBuffer<T>);
  fn store_local_param(&mut self, param_writer: &mut WriteBuffer<T>);
  fn store_global_param(&mut self, param_writer: &mut WriteBuffer<T>);*/

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
