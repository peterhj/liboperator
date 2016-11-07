//#![feature(associated_type_defaults)]
//#![feature(conservative_impl_trait)]
#![feature(reflect_marker)]
#![feature(zero_one)]

extern crate csv;
extern crate densearray;
extern crate rng;
extern crate sharedmem;
extern crate typemap;

extern crate rand;
extern crate rustc_serialize;

//use io::{IoBuffer};
use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::cell::{Cell, RefCell};
use std::io::{Read, Write};
use std::ops::{Deref};
use std::rc::{Rc};

pub mod data;
pub mod io;
pub mod opt;
pub mod prelude;
pub mod rw;
pub mod timing;

thread_local! {
  static OP_NODE_ID_COUNTER: Cell<u16> = Cell::new(0);
}

#[derive(Clone, Copy)]
pub enum OpCapability {
  Forward,
  Backward,
  RForward,
  RBackward,
}

impl OpCapability {
  pub fn enable_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      _ => true,
    }
  }

  pub fn enable_r_forward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      _ => true,
    }
  }

  pub fn enable_r_backward(&self) -> bool {
    match *self {
      OpCapability::Forward => false,
      OpCapability::Backward => false,
      OpCapability::RForward => false,
      _ => true,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpPhase {
  Inference,
  Learning,
}

#[derive(Clone, Copy, Debug)]
pub enum Regularization {
  L2(f32),
}

pub struct OperatorStackEntry {
  epoch:    u64,
  count:    u64,
}

#[derive(Default)]
pub struct OperatorStack {
  curr_epoch:   Cell<u64>,
  entries:      RefCell<Vec<OperatorStackEntry>>,
}

impl OperatorStack {
  pub fn _next(&self) -> u64 {
    if 0 == self.curr_epoch.get() {
      OP_NODE_ID_COUNTER.with(|op_node_id_ctr| {
        op_node_id_ctr.set(op_node_id_ctr.get() + 1);
        let node_id = op_node_id_ctr.get();
        assert!(node_id != 0);
        self.curr_epoch.set(node_id as u64);
      });
    }
    self.curr_epoch.set(self.curr_epoch.get() + 0x10000);
    self.curr_epoch.get()
  }

  pub fn _epoch(&self) -> u64 {
    unimplemented!();
  }

  pub fn count(&self) -> u64 {
    let entries = self.entries.borrow();
    assert!(!entries.is_empty());
    entries.last().unwrap().count
  }

  pub fn limit(&self, max_count: u64) -> bool {
    let entries = self.entries.borrow();
    assert!(!entries.is_empty());
    entries.last().unwrap().count <= max_count
  }

  pub fn push(&self, epoch: u64) {
    let mut entries = self.entries.borrow_mut();
    if entries.len() == 10 {
      println!("WARNING: operator stack depth is 10, probably a bug!");
    }
    if !entries.is_empty() && epoch == entries.last().unwrap().epoch {
      entries.last_mut().unwrap().count += 1;
    } else {
      entries.push(OperatorStackEntry{
        epoch:  epoch,
        count:  1,
      });
    }
  }

  pub fn pop(&self, epoch: u64) {
    let mut entries = self.entries.borrow_mut();
    assert!(!entries.is_empty());
    assert_eq!(epoch, entries.last().unwrap().epoch);
    entries.last_mut().unwrap().count -= 1;
    if 0 == entries.last().unwrap().count {
      entries.pop();
    }
  }
}

pub trait Operator {
  fn _next(&self) -> u64;
  fn _epoch(&self) -> u64;
}

#[derive(Clone, Copy, Default)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, Default)]
pub struct Epoch {
  pub node_id:  NodeId,
  pub epoch_nr: u64,
}

#[derive(Clone, Default)]
pub struct OperatorNode {
  pub curr_epoch:   Cell<u64>,
  //pub node_id:      Cell<Option<NodeId>>,
  //pub curr_epoch:   Cell<Epoch>,
  pub curr_count:   Cell<u64>,
}

impl Operator for OperatorNode {
  fn _next(&self) -> u64 {
    if 0 == self.curr_epoch.get() {
      OP_NODE_ID_COUNTER.with(|op_node_id_ctr| {
        op_node_id_ctr.set(op_node_id_ctr.get() + 1);
        let node_id = op_node_id_ctr.get();
        assert!(node_id != 0);
        self.curr_epoch.set(node_id as u64);
      });
    }
    self.curr_epoch.set(self.curr_epoch.get() + 0x10000);
    self.curr_count.set(0);
    self.curr_epoch.get()
  }

  fn _epoch(&self) -> u64 {
    self.curr_epoch.get()
  }
}

impl OperatorNode {
  pub fn step(&self, next_epoch: u64) {
    assert!(next_epoch >= self.curr_epoch.get());
    if next_epoch != self.curr_epoch.get() {
      self.curr_epoch.set(next_epoch);
      self.curr_count.set(0);
    }
    self.curr_count.set(self.curr_count.get() + 1);
  }

  pub fn count(&self) -> u64 {
    self.curr_count.get()
  }

  pub fn limit(&self, max_count: u64) -> bool {
    self.curr_count.get() <= max_count
  }
}

/*pub trait OperatorOutput {
}*/

pub trait NewDiffOperator<S>: Operator {
  type IoBuf: ?Sized;
  //type OpRef = Rc<RefCell<NewDiffOperator<S, IoBuf=Self::IoBuf, OpRef=Self::OpRef>>>;

  fn _traverse_fwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>));
  fn _traverse_bwd(&mut self, _epoch: u64, _apply: &mut FnMut(&mut NewDiffOperator<S, IoBuf=Self::IoBuf>));
  //fn _traverse_fwd_new(&self, _epoch: u64, _apply: &mut FnMut(Self::OpRef));
  //fn _traverse_bwd_new(&self, _epoch: u64, _apply: &mut FnMut(Self::OpRef));

  fn _diff_param_sz(&self) -> usize { 0 }
  fn _nondiff_param_sz(&self) -> usize { 0 }

  fn _load_diff_param(&mut self, offset: usize, param_reader: &mut Self::IoBuf) -> usize { 0 }
  fn _load_nondiff_param(&mut self, offset: usize, param_reader: &mut Self::IoBuf) -> usize { 0 }
  fn _store_diff_param(&mut self, offset: usize, param_writer: &mut Self::IoBuf) -> usize { 0 }
  fn _store_nondiff_param(&mut self, offset: usize, param_writer: &mut Self::IoBuf) -> usize { 0 }
  fn _store_grad(&mut self, offset: usize, grad_writer: &mut Self::IoBuf) -> usize { 0 }

  fn _save_rng_state(&mut self) {}
  fn _restore_rng_state(&mut self) {}

  fn _next_iteration(&mut self) {}
  fn _load_batch(&mut self, _samples: &[S]) {}
  fn _init_param(&mut self, rng: &mut Xorshiftplus128Rng) {}
  fn _update_nondiff_param(&mut self, _iter_nr: usize) {}
  fn _reset_grad(&mut self) {}

  fn _forward(&mut self, phase: OpPhase);
  fn _backward(&mut self);

  fn _dump_input(&mut self) -> Vec<u8> { unimplemented!(); }
  fn _dump_output(&mut self) -> Vec<u8> { unimplemented!(); }
}

pub trait DiffOperatorRma<S, RmaBuf>: NewDiffOperator<S> {
  type Ctx;

  fn _rma_load_diff_param(&mut self, offset: usize, param_reader: &mut RmaBuf, ctx: Self::Ctx) -> usize { 0 }
  fn _rma_load_nondiff_param(&mut self, offset: usize, param_reader: &mut RmaBuf) -> usize { 0 }
  fn _rma_store_diff_param(&mut self, offset: usize, param_writer: &mut RmaBuf, ctx: Self::Ctx) -> usize { 0 }
  fn _rma_store_nondiff_param(&mut self, offset: usize, param_writer: &mut RmaBuf) -> usize { 0 }
  fn _rma_store_grad(&mut self, offset: usize, grad_writer: &mut RmaBuf, ctx: Self::Ctx) -> usize { 0 }
}

pub trait DiffLoss<S>: NewDiffOperator<S> {
  fn reset_loss(&mut self);
  fn store_loss(&mut self) -> f32;
  fn _store_accuracy(&mut self) -> usize { 0 }
  fn _get_pred(&mut self) -> &[f32] { unimplemented!(); }
  fn _get_target(&mut self) -> &[f32] { unimplemented!(); }
  fn _get_delta(&mut self) -> &[f32] { unimplemented!(); }

  fn diff_param_sz(&mut self) -> usize {
    let epoch = self._next();
    let mut grad_sz = 0;
    self._traverse_bwd(epoch, &mut |op| {
      grad_sz += op._diff_param_sz();
    });
    grad_sz
  }

  fn nondiff_param_sz(&mut self) -> usize {
    let epoch = self._next();
    let mut nondiff_sz = 0;
    self._traverse_bwd(epoch, &mut |op| {
      nondiff_sz += op._nondiff_param_sz();
    });
    nondiff_sz
  }

  fn load_diff_param(&mut self, param_reader: &mut Self::IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_diff_param(offset, param_reader);
    });
    offset
  }

  fn load_nondiff_param(&mut self, param_reader: &mut Self::IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_nondiff_param(offset, param_reader);
    });
    offset
  }

  fn store_diff_param(&mut self, param_writer: &mut Self::IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_diff_param(offset, param_writer);
    });
    offset
  }

  fn store_nondiff_param(&mut self, param_writer: &mut Self::IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_nondiff_param(offset, param_writer);
    });
    offset
  }

  fn store_grad(&mut self, grad_writer: &mut Self::IoBuf) -> usize {
    //grad_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_grad(offset, grad_writer);
    });
    offset
  }

  fn next_iteration(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._next_iteration());
  }

  fn load_batch(&mut self, samples: &[S]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._load_batch(samples));
  }

  fn save_rng_state(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._save_rng_state());
  }

  fn restore_rng_state(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._restore_rng_state());
  }

  fn init_param(&mut self, rng: &mut Xorshiftplus128Rng) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._init_param(rng));
  }

  fn reset_nondiff_param(&mut self, iter_nr: usize) {
    let epoch = self._next();
    // FIXME(20161020)
    unimplemented!();
    //self._traverse_bwd(epoch, &mut |op| op._update_nondiff_param(iter_nr));
  }

  fn update_nondiff_param(&mut self, iter_nr: usize) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._update_nondiff_param(iter_nr));
  }

  fn reset_grad(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._reset_grad());
  }

  fn forward(&mut self, phase: OpPhase) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._forward(phase));
  }

  fn backward(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._backward());
  }
}

pub trait LossReport<Stats> {
  fn update_stats(&mut self, iter_nr: usize, stats: &mut Stats);
}

#[derive(Clone, Copy, Default, Debug, RustcEncodable)]
pub struct ClassLossRecord {
  pub iter:     usize,
  pub loss:     f32,
  pub accuracy: f32,
  pub elapsed:  f64,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ClassLossStats {
  pub iter_nr:          usize,
  pub sample_count:     usize,
  pub correct_count:    usize,
  pub accum_loss:       f32,
}

impl ClassLossStats {
  pub fn reset(&mut self) {
    self.iter_nr = 0;
    self.sample_count = 0;
    self.correct_count = 0;
    self.accum_loss = 0.0;
  }

  pub fn to_record(&self, elapsed: f64) -> ClassLossRecord {
    ClassLossRecord{
      iter:     self.iter_nr,
      loss:     self.avg_loss(),
      accuracy: self.accuracy(),
      elapsed:  elapsed,
    }
  }

  pub fn avg_loss(&self) -> f32 {
    self.accum_loss / self.sample_count as f32
  }

  pub fn accuracy(&self) -> f32 {
    self.correct_count as f32 / self.sample_count as f32
  }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RegressLossStats {
  pub iter_nr:          usize,
  pub sample_count:     usize,
  pub avg_loss:         f32,
}

impl RegressLossStats {
  pub fn reset(&mut self) {
    self.iter_nr = 0;
    self.sample_count = 0;
    self.avg_loss = 0.0;
  }
}

pub trait DiffOperator<T> {
  type Output;
  type Rng: Rng;

  fn _output(&self, arm: usize) -> Self::Output;

  fn diff_param_sz(&self) -> usize { 0 }
  fn nondiff_param_sz(&self) -> usize { 0 }
  fn total_param_sz(&self) -> usize {
    self.diff_param_sz() + self.nondiff_param_sz()
  }

  fn save_rng_state(&mut self) {}
  fn restore_rng_state(&mut self) {}
  //fn store_rng_state(&mut self, _rng_state: &mut Write) {}
  //fn load_rng_state(&mut self, _rng_state: &mut Read) {}

  //fn reset_state(&mut self) {}

  fn init_param(&mut self, _rng: &mut Self::Rng) {}
  //fn reset_nondiff_param(&mut self) {}
  fn reset_grad(&mut self) {}

  fn store_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn store_nondiff_param(&mut self, _param_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn load_nondiff_param(&mut self, _param_reader: &mut ReadBuffer<T>, _offset: usize) -> usize { 0 }
  fn update_param(&mut self, _alpha: f32, _beta: f32, _grad_reader: &mut ReadAccumulateBuffer<T>, _offset: usize) -> usize { 0 }

  fn update_nondiff_param(&mut self, _iter: usize) {}

  fn store_grad(&mut self, _grad_writer: &mut WriteBuffer<T>, _offset: usize) -> usize { 0 }
  fn accumulate_grad(&mut self, _alpha: f32, _beta: f32, _grad_accum: &mut AccumulateBuffer<T>, _offset: usize) -> usize { 0 }
  //fn grad_step(&mut self, _alpha: f32, _beta: f32) {}

  fn reset_loss(&mut self) {}
  fn store_loss(&mut self) -> f32 { 0.0 }
  fn _store_accuracy(&mut self) -> usize { 0 }

  fn apply_grad_reg(&mut self, _reg: Regularization) {}

  fn forward(&mut self, phase: OpPhase);
  //fn fwd_reg(&mut self, _reg: Regularization) {}
  fn backward(&mut self);
  //fn bwd_reg(&mut self, _reg: Regularization) {}
  //fn r_forward(&mut self) { unimplemented!(); }
  //fn r_backward(&mut self) { unimplemented!(); }
}

pub trait DiffOperatorIo<T, Buf>/*: DiffOperator*/ where T: Copy {
  fn store_diff_param(&mut self, _param_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn store_nondiff_param(&mut self, _param_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn load_diff_param(&mut self, _param_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn load_nondiff_param(&mut self, _param_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn update_diff_param(&mut self, _alpha: T, _beta: T, _delta_reader: &mut Buf, _offset: usize) -> usize { 0 }
  fn store_grad(&mut self, _grad_writer: &mut Buf, _offset: usize) -> usize { 0 }
  fn accumulate_grad(&mut self, _alpha: T, _beta: T, _grad_writer: &mut Buf, _offset: usize) -> usize { 0 }
}

pub trait DiffOperatorInput<T, S>: DiffOperator<T> {
  fn as_op(&self) -> &DiffOperator<T, Output=Self::Output, Rng=Self::Rng> { unimplemented!(); }
  fn load_data(&mut self, samples: &[S]);
}

pub trait DiffOperatorOutput<T, U>: DiffOperator<T> {
  fn get_output(&mut self) -> U;
}

pub trait CheckpointFormat {
}

pub trait DiffOperatorCheckpoint<Format> where Format: CheckpointFormat {
  fn decode(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn encode(&mut self, writer: &mut Write) -> Result<(), ()>;
}
