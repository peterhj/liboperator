#[derive(Clone, Debug)]
pub enum Shape {
  Shape1d(usize),
  Shape2d((usize, usize)),
  Shape3d((usize, usize, usize)),
  Shape4d((usize, usize, usize, usize)),
  //ShapeNd(Vec<usize>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeDesc {
  Channel(usize),
  Width,
  Height,
  Depth,
  Time,
  Frequency,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ClassLoss;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct RegressLoss;

pub trait SampleDatum<U: ?Sized> {
  fn extract_input(&self, _dst: &mut U) -> Result<(), ()> { Err(()) }
  fn shape(&self) -> Option<Shape> { None }
  fn shape_desc(&self) -> Option<ShapeDesc> { None }
}

pub trait SampleLabel {
  fn class(&self) -> Option<u32> { None }
  fn target(&self) -> Option<f32> { None }
}

pub trait SampleLossWeight<A> {
  fn weight(&self) -> Option<f32> { None }
  fn mix_weight(&mut self, w: f32) -> Result<(), ()> { Err(()) }
}
