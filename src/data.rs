// XXX(20161008): `SampleDatum`, `SampleLabel`, and `SampleLossWeight` are the new traits.

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ClassLoss;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct RegressLoss;

pub trait SampleDatum<U: ?Sized> {
  fn extract_input(&self, _dst: &mut U) -> Result<(), ()> { Err(()) }
}

pub trait SampleLabel {
  fn class(&self) -> Option<u32> { None }
  fn target(&self) -> Option<f32> { None }
}

pub trait SampleLossWeight<A> {
  fn weight(&self) -> Option<f32> { None }
  fn mix_weight(&mut self, w: f32) -> Result<(), ()> { Err(()) }
}

/*#[deprecated]
pub trait SampleInput<T> {
  fn input(&self) -> &[T];
}

pub trait SampleExtractInput<U> {
  fn extract_input(&self, dst: &mut [U]);
}

pub trait SampleClass {
  fn class(&self) -> Option<u32>;
  //fn class_weight(&self) -> Option<f32>;
  //fn mix_class_weight(&mut self, w: f32);
}

pub trait SampleScalarTarget<U> {
  fn scalar_target(&self) -> Option<U>;
  fn scalar_target_weight(&self) -> Option<f32>;
}

pub trait SampleWeight {
  fn weight(&self) -> Option<f32>;
  fn mix_weight(&mut self, w: f32);
}*/
