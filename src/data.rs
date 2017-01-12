use densearray::prelude::*;
use sharedmem::{SharedSlice};
use typemap::{ShareMap, TypeMap, Key};

use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait SampleExtractInput<U: ?Sized> {
  fn extract_input(&self, output: &mut U) -> Result<usize, ()>;
  fn parallel_extract_input(&self, output: &mut U) -> Result<usize, ()> { unimplemented!(); }
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

  fn parallel_extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output.reshape_mut(len).parallel_cast(self.reshape(len));
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

  fn parallel_extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output.reshape_mut(len).parallel_copy(self.reshape(len));
    Ok(len)
  }
}

/*pub trait SampleParallelExtractInput<U: ?Sized>: Send + Sync {
  fn parallel_extract_input(&self, output: &mut U) -> Result<usize, ()>;
}

impl SampleParallelExtractInput<[f32]> for SharedSlice<u8> {
  fn parallel_extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    output.reshape_mut(len).parallel_cast(self.reshape(len));
    Ok(len)
  }
}

impl SampleParallelExtractInput<[f32]> for SharedSlice<f32> {
  fn parallel_extract_input(&self, output: &mut [f32]) -> Result<usize, ()> {
    let len = self.len();
    assert!(len <= output.len());
    //output[ .. len].copy_from_slice(&*self);
    output.reshape_mut(len).parallel_copy(self.reshape(len));
    Ok(len)
  }
}

pub trait SampleExtractTaggedInput<U: ?Sized>: Send + Sync {
  fn extract_tagged_input(&self, tag: usize, output: &mut U) -> Result<usize, ()>;
}*/

pub trait SampleInputShape<Shape> where Shape: PartialEq + Eq {
  fn input_shape(&self) -> Option<Shape>;
}

impl SampleInputShape<(usize, usize, usize)> for (usize, usize, usize) {
  fn input_shape(&self) -> Option<(usize, usize, usize)> {
    Some(*self)
  }
}

/*pub trait SharedSampleInputShape<Shape>: Send + Sync where Shape: PartialEq + Eq {
  fn input_shape(&self) -> Option<Shape>;
}

impl SharedSampleInputShape<(usize, usize, usize)> for (usize, usize, usize) {
  fn input_shape(&self) -> Option<(usize, usize, usize)> {
    Some(*self)
  }
}*/

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

pub struct SharedSampleItem {
  pub kvs:  ShareMap,
}

impl SharedSampleItem {
  pub fn new() -> SharedSampleItem {
    SharedSampleItem{
      kvs:  TypeMap::custom(),
    }
  }
}

pub struct SampleSharedSliceDataKey<T> where T: 'static + Copy {
  _marker:  PhantomData<T>,
}

impl<T> Key for SampleSharedSliceDataKey<T> where T: 'static + Copy {
  type Value = SharedSlice<T>;
}

pub struct SampleExtractInputKey<U: ?Sized> where U: 'static {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SampleExtractInputKey<U> where U: 'static {
  type Value = Rc<SampleExtractInput<U>>;
  //type Value = Arc<SampleExtractInput<U>>;
}

pub struct SharedSampleExtractInputKey<U: ?Sized> where U: 'static {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SharedSampleExtractInputKey<U> where U: 'static {
  type Value = Arc<SampleExtractInput<U> + Send + Sync>;
}

pub struct SampleSharedExtractInputKey<U: ?Sized> where U: 'static {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SampleSharedExtractInputKey<U> where U: 'static {
  type Value = Arc<SampleExtractInput<U> + Send + Sync>;
}

/*pub struct SharedSampleParallelExtractInputKey<U: ?Sized> where U: 'static {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SharedSampleParallelExtractInputKey<U> where U: 'static {
  type Value = Arc<SampleParallelExtractInput<U>>;
}

pub struct SampleExtractTaggedInputKey<U: ?Sized> where U: 'static {
  _marker:  PhantomData<U>,
}

impl<U: ?Sized> Key for SampleExtractTaggedInputKey<U> where U: 'static {
  type Value = Rc<SampleExtractTaggedInput<U>>;
  //type Value = Arc<SampleExtractTaggedInput<U>>;
}*/

pub struct SampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  _marker:  PhantomData<Shape>,
}

impl<Shape> Key for SampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  type Value = Rc<SampleInputShape<Shape>>;
  //type Value = Arc<SampleInputShape<Shape>>;
}

pub struct SharedSampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  _marker:  PhantomData<Shape>,
}

impl<Shape> Key for SharedSampleInputShapeKey<Shape> where Shape: 'static + PartialEq + Eq {
  type Value = Arc<SampleInputShape<Shape> + Send + Sync>;
}

/*pub struct SampleInputShape3dKey {}

impl Key for SampleInputShape3dKey {
  type Value = (usize, usize, usize);
}*/

pub struct SampleInputTagsKey {}

impl Key for SampleInputTagsKey {
  type Value = usize;
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
