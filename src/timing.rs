use std::time::{Instant};

pub fn instant_diff_seconds(lap_time: Instant, start_time: Instant) -> f64 {
  let elapsed_dur = lap_time - start_time;
  let elapsed_s = elapsed_dur.as_secs() as f64;
  let elapsed_ns = elapsed_dur.subsec_nanos() as f64;
  let elapsed = elapsed_s + elapsed_ns * 1.0e-9;
  elapsed
}

pub struct Stopwatch {
  prev_time:    Instant,
  lap_time:     Instant,
}

impl Stopwatch {
  pub fn new() -> Self {
    let now = Instant::now();
    Stopwatch{
      prev_time:    now,
      lap_time:     now,
    }
  }

  pub fn lap(&mut self) -> &mut Stopwatch{
    self.prev_time = self.lap_time;
    self.lap_time = Instant::now();
    self
  }

  pub fn elapsed(&self) -> f64 {
    instant_diff_seconds(self.lap_time, self.prev_time)
  }
}
