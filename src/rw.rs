use std::cmp::{min};

pub trait ReadBuffer<T> where T: Copy {
  fn read(&mut self, offset: usize, dst: &mut [T]) -> usize;
}

pub trait WriteBuffer<T> where T: Copy {
  fn write(&mut self, offset: usize, src: &[T]) -> usize;
}

pub trait AccumulateBuffer<T> where T: Copy {
  fn accumulate(&mut self, alpha: T, beta: T, offset: usize, src: &[T]) -> usize;
}

impl<T> ReadBuffer<T> for Vec<T> where T: Copy {
  fn read(&mut self, offset: usize, dst: &mut [T]) -> usize {
    assert!(offset <= self.len());
    let copy_len = min(self.len() - offset, dst.len());
    dst[ .. copy_len].copy_from_slice(&self[offset .. offset + copy_len]);
    copy_len
  }
}

impl<T> WriteBuffer<T> for Vec<T> where T: Copy {
  fn write(&mut self, offset: usize, src: &[T]) -> usize {
    assert!(offset <= self.len());
    let copy_len = min(self.len() - offset, src.len());
    self[offset .. offset + copy_len].copy_from_slice(&src[ .. copy_len]);
    copy_len
  }
}

impl AccumulateBuffer<f32> for Vec<f32> {
  fn accumulate(&mut self, alpha: f32, beta: f32, offset: usize, src: &[f32]) -> usize {
    assert!(offset <= self.len());
    let copy_len = min(self.len() - offset, src.len());
    for i in 0 .. copy_len {
      let x = src[i];
      let y = self[offset + i];
      self[offset + i] = alpha * x + beta * y;
    }
    copy_len
  }
}
