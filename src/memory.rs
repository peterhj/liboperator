use std::ops::{Deref};
use std::rc::{Rc};
use std::slice::{from_raw_parts};

pub trait SharedBuffer<T>: Drop {
  fn as_ptr_shared(&self) -> *const T;
  fn len_shared(&self) -> usize;
}

pub struct SharedMem<T> {
  buf:  Rc<Box<SharedBuffer<T>>>,
}

impl<T> SharedMem<T> {
  pub fn new<Buf>(buf: Buf) -> SharedMem<T> where Buf: SharedBuffer<T> {
    let buf: Box<SharedBuffer<T>> = Box::new(buf);
    unimplemented!();
  }

  pub fn as_slice(&self) -> SharedSlice<T> {
    SharedSlice{
      buf:  self.buf.clone(),
      ptr:  self.buf.as_ptr_shared(),
      len:  self.buf.len_shared(),
    }
  }

  pub fn slice(&self, from_idx: usize, to_idx: usize) -> SharedSlice<T> {
    assert!(from_idx < self.buf.len_shared());
    assert!(to_idx - from_idx < self.buf.len_shared());
    SharedSlice{
      buf:  self.buf.clone(),
      ptr:  unsafe { self.buf.as_ptr_shared().offset(from_idx as isize) },
      len:  to_idx - from_idx,
    }
  }
}

pub struct SharedSlice<T> {
  ptr:  *const T,
  len:  usize,
  buf:  Rc<Box<SharedBuffer<T>>>,
}

impl<T> Deref for SharedSlice<T> {
  type Target = [T];

  fn deref(&self) -> &[T] {
    unsafe { from_raw_parts(self.ptr, self.len) }
  }
}
