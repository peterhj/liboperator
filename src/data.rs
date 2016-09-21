pub trait SampleInput<T> {
  fn input(&self) -> &[T];
}

pub trait SampleExtractInput<U> {
  fn extract_input(&self, dst: &mut [U]);
}

pub trait SampleClass {
  fn class(&self) -> Option<u32>;
}

pub trait SampleWeight {
  fn weight(&self) -> Option<f32>;
  fn mix_weight(&mut self, w: f32);
}
