use std::cmp::{min};

/*pub trait IoBuffer<Scalar, Target: ?Sized>: ?Sized where Scalar: Copy {
  fn read(&mut self, offset: usize, dst: &mut Target) -> usize;
  fn read_accumulate(&mut self, alpha: Scalar, beta: Scalar, offset: usize, dst: &mut Target) -> usize;
  fn write(&mut self, offset: usize, src: &Target) -> usize;
  fn accumulate(&mut self, alpha: Scalar, beta: Scalar, offset: usize, src: &Target) -> usize;
}*/

pub trait IoBuffer<Target: ?Sized> {
  fn read_buf(&mut self, offset: usize, dst: &mut Target) -> usize;
  fn write_buf(&mut self, offset: usize, src: &Target) -> usize;
}

//impl IoBuffer<f32, [f32]> for [f32] {
impl IoBuffer<[f32]> for [f32] {
  fn read_buf(&mut self, offset: usize, dst: &mut [f32]) -> usize {
    assert!(offset <= self.len());
    //let copy_len = min(self.len() - offset, dst.len());
    let copy_len = dst.len();
    assert!(offset + copy_len <= self.len());
    dst[ .. copy_len].copy_from_slice(&self[offset .. offset + copy_len]);
    copy_len
  }

  /*fn read_accumulate(&mut self, alpha: f32, beta: f32, offset: usize, dst: &mut [f32]) -> usize {
    assert!(offset <= self.len());
    let copy_len = min(self.len() - offset, dst.len());
    for i in 0 .. copy_len {
      let x = self[offset + i];
      let y = dst[i];
      dst[i] = alpha * x + beta * y;
    }
    copy_len
  }*/

  fn write_buf(&mut self, offset: usize, src: &[f32]) -> usize {
    assert!(offset <= self.len());
    //let copy_len = min(self.len() - offset, src.len());
    let copy_len = src.len();
    assert!(offset + copy_len <= self.len());
    self[offset .. offset + copy_len].copy_from_slice(&src[ .. copy_len]);
    copy_len
  }

  /*fn accumulate(&mut self, alpha: f32, beta: f32, offset: usize, src: &[f32]) -> usize {
    assert!(offset <= self.len());
    let copy_len = min(self.len() - offset, src.len());
    for i in 0 .. copy_len {
      let x = src[i];
      let y = self[offset + i];
      self[offset + i] = alpha * x + beta * y;
    }
    copy_len
  }*/
}
