use sharedmem::{SharedSlice};
use typemap::{TypeMap, Key};

use std::marker::{PhantomData, Reflect};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait SampleExtractInput<U: ?Sized> {
  fn extract_input(&self, output: &mut U) -> Result<usize, ()>;
}

impl SampleExtractInput<[u8]> for Vec<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output[ .. len].copy_from_slice(self);
    Ok(len)
  }
}

impl SampleExtractInput<[f32]> for Vec<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    for (x, y) in self.iter().zip(output[ .. len].iter_mut()) {
      *y = *x as f32;
    }
    Ok(len)
  }
}

impl SampleExtractInput<[f32]> for Vec<f32> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output[ .. len].copy_from_slice(self);
    Ok(len)
  }
}

impl SampleExtractInput<[u8]> for SharedSlice<u8> {
  fn extract_input(&self, output: &mut [u8]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output[ .. len].copy_from_slice(&*self);
    Ok(len)
  }
}

impl SampleExtractInput<[f32]> for SharedSlice<u8> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    for (x, y) in (*self).iter().zip(output[ .. len].iter_mut()) {
      *y = *x as f32;
    }
    Ok(len)
  }
}

impl SampleExtractInput<[f32]> for SharedSlice<f32> {
  fn extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output[ .. len].copy_from_slice(&*self);
    Ok(len)
  }
}

pub trait SampleInputShape<Shape> where Shape: PartialEq + Eq {
  fn input_shape(&self) -> Option<Shape>;
}

impl SampleInputShape<(usize, usize, usize)> for (usize, usize, usize) {
  fn input_shape(&self) -> Option<(usize, usize, usize)> {
    Some(*self)
  }
}

pub struct SampleItem {
  pub kvs:  TypeMap,
}

impl SampleItem {
  pub fn new() -> SampleItem {
    SampleItem{
      kvs:  TypeMap::new(),
    }
  }
}

pub struct SampleSharedSliceDataKey<T> where T: 'static + Copy + Reflect {
  _marker:  PhantomData<T>,
}

impl<T> Key for SampleSharedSliceDataKey<T> where T: 'static + Copy + Reflect {
  type Value = SharedSlice<T>;
}

pub struct SampleExtractInputKey<U: ?Sized> where U: 'static + Reflect {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SampleExtractInputKey<U> where U: 'static + Reflect {
  type Value = Rc<SampleExtractInput<U>>;
}

pub struct SampleSharedExtractInputKey<U: ?Sized> where U: 'static + Reflect {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SampleSharedExtractInputKey<U> where U: 'static + Reflect {
  type Value = Arc<SampleExtractInput<U>>;
}

pub struct SampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  _marker:  PhantomData<Shape>,
}

impl<Shape> Key for SampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  type Value = Rc<SampleInputShape<Shape>>;
}

pub struct SampleSharedInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  _marker:  PhantomData<Shape>,
}

impl<Shape> Key for SampleSharedInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  type Value = Arc<SampleInputShape<Shape>>;
}

pub struct SampleInputShape3dKey {}

impl Key for SampleInputShape3dKey {
  type Value = (usize, usize, usize);
}

pub struct SampleClassLabelKey {}

impl Key for SampleClassLabelKey {
  type Value = u32;
}

pub struct SampleRegressTargetKey {}

impl Key for SampleRegressTargetKey {
  type Value = f32;
}

pub struct SampleWeightKey {}

impl Key for SampleWeightKey {
  type Value = f32;
}

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

/*#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleKey {
  Input,
  ClassLabel,
  RegressTarget,
  Weight,
}*/

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ClassLoss;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct RegressLoss;

pub trait SampleDatum<U: ?Sized> {
  fn input(&self) -> Option<&U> { None }
  fn extract_input(&self, _dst: &mut U) -> Result<(), ()> { Err(()) }
  fn len(&self) -> Option<usize> { None }
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
