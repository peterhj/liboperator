use std::time::{Instant};

pub fn instant_diff_seconds(lap_time: Instant, start_time: Instant) -> f64 {
  let elapsed_dur = lap_time - start_time;
  let elapsed_s = elapsed_dur.as_secs() as f64;
  let elapsed_ns = elapsed_dur.subsec_nanos() as f64;
  let elapsed = elapsed_s + elapsed_ns * 1.0e-9;
  elapsed
}
