//use array::{Array2d, Array3d};

pub trait SampleCastAs<Target=Self> {
}

pub struct ClassSampleU8 {
  pub input:    Vec<u8>,
  pub label:    Option<u32>,
  pub weight:   f32,
}

impl SampleCastAs for ClassSampleU8 {}
impl SampleCastAs<ClassSampleF32> for ClassSampleU8 {}

pub struct ClassSampleF32 {
  pub input:    Vec<f32>,
  pub label:    Option<u32>,
  pub weight:   f32,
}

impl SampleCastAs for ClassSampleF32 {}

/*pub enum DynamicNdBuf<T> where T: Copy {
  Buf(Vec<T>),
  Dim1(Array2d<T>),
  Dim2(Array3d<T>),
}

pub struct ClassSample<T> where T: Copy {
  pub input:    Vec<T>,
  pub label:    Option<i32>,
}

pub struct SparseClassSample<T> where T: Copy {
  pub input:    Vec<(u32, T)>,
  pub label:    Option<i32>,
}

pub struct ClassSample1d<T> where T: Copy {
  pub input:    Array2d<T>,
  pub label:    Option<i32>,
}

pub struct ClassSample2d<T> where T: Copy {
  pub input:    Array3d<T>,
  pub label:    Option<i32>,
}

pub struct RegressSample<T> where T: Copy {
  pub input:    Vec<T>,
  pub label:    Vec<f32>,
}

pub struct AutoRegressSample<T> where T: Copy {
  pub input:    Vec<T>,
  pub label:    Vec<T>,
}*/
