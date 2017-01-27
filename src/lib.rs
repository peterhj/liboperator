//#![feature(associated_type_defaults)]
//#![feature(conservative_impl_trait)]
//#![feature(reflect_marker)]
#![feature(fn_traits)]
#![feature(integer_atomics)]
#![feature(unboxed_closures)]
#![feature(zero_one)]

extern crate csv;
extern crate densearray;
extern crate rng;
extern crate sharedmem;
extern crate typemap_alt as typemap;

//extern crate lazy_static;
extern crate rand;
extern crate rustc_serialize;

//use io::{IoBuffer};
//use rw::{ReadBuffer, ReadAccumulateBuffer, WriteBuffer, AccumulateBuffer};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng};
use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::io::{Read, Write};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};
use std::sync::atomic::{ATOMIC_U64_INIT, AtomicU64, Ordering};

pub mod data;
pub mod io;
pub mod opt;
pub mod prelude;
pub mod rw;
pub mod timing;

thread_local! {
  static OP_NODE_ID_COUNTER: Cell<u16> = Cell::new(0);
}

static NODE_ID_COUNTER:     AtomicU64 = ATOMIC_U64_INIT;
static EPOCH_NR_COUNTER:    AtomicU64 = ATOMIC_U64_INIT;
static BATCH_NR_COUNTER:    AtomicU64 = ATOMIC_U64_INIT;

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

pub trait Operator {
  fn _next(&self) -> u64;
  fn _epoch(&self) -> u64 { unimplemented!(); }
}

/*#[derive(Clone, Copy, Default)]
pub struct NodeId(pub u32);*/

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl Default for NodeId {
  fn default() -> NodeId {
    NodeId(0)
  }
}

impl NodeId {
  pub fn new() -> NodeId {
    let node_id = NODE_ID_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    assert!(node_id != 0);
    NodeId(node_id)
  }
}

#[derive(Clone, Copy)]
pub struct EpochNr(u64);

impl Default for EpochNr {
  fn default() -> EpochNr {
    EpochNr(0)
  }
}

impl EpochNr {
  pub fn new() -> EpochNr {
    let epoch_nr = EPOCH_NR_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    assert!(epoch_nr != 0);
    EpochNr(epoch_nr)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchNr(u64);

impl Default for BatchNr {
  fn default() -> BatchNr {
    BatchNr(0)
  }
}

impl BatchNr {
  pub fn new() -> BatchNr {
    let batch_nr = BATCH_NR_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    assert!(batch_nr != 0);
    BatchNr(batch_nr)
  }
}

/*pub struct BatchNrSet(HashSet<BatchNr>);

impl BatchNrSet {
  pub fn new() -> BatchNrSet {
    BatchNrSet(HashSet::new())
  }
}*/

#[derive(Clone, Copy, Default)]
pub struct Epoch {
  pub epoch_nr: EpochNr,
  pub root:     NodeId,
}

pub struct NodeStackEntry {
  epoch:    u64,
  //epoch:    Epoch,
  count:    u64,
}

//#[derive(Default)]
pub struct NodeStack {
  node_id:      NodeId,
  curr_epoch:   Cell<u64>,
  //curr_epoch:   Cell<Epoch>,
  entries:      RefCell<Vec<NodeStackEntry>>,
}

impl Default for NodeStack {
  fn default() -> NodeStack {
    let node_id = NodeId::new();
    NodeStack{
      node_id:      node_id,
      curr_epoch:   Cell::new(0),
      //curr_epoch:   Cell::new(Epoch::default()),
      entries:      RefCell::new(vec![]),
    }
  }
}

impl NodeStack {
  /// Initiate a new (acyclic) traversal starting from the current node.
  /// The node needs to increment the epoch number counter.
  /// The traversal epoch is uniquely identified by the current node ID and the
  /// the (incremented) epoch number.
  ///
  /// TODO(20170112):
  /// Currently the epoch is a hack: take a 16-bit node ID and a 48-bit epoch
  /// number and OR them together. A better `Epoch` uses two separate 64-bit
  /// words for each.
  ///
  /// Failure modes:
  /// - Node ID is an invalid value.
  /// - Epoch number is an invalid value.
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

  pub fn epoch(&self) -> u64 {
    let entries = self.entries.borrow();
    assert!(!entries.is_empty());
    entries.last().unwrap().epoch
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
    // FIXME(20170112): should check to make sure the traversal is acyclic.
    if !entries.is_empty() && epoch == entries.last().unwrap().epoch {
      entries.last_mut().unwrap().count += 1;
    } else {
      entries.push(NodeStackEntry{
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

// XXX(20170112): `NodeCell` is deprecated; use `NodeStack` instead.
#[derive(Clone, Default)]
pub struct NodeCell {
  pub curr_epoch:   Cell<u64>,
  //pub node_id:      Cell<Option<NodeId>>,
  //pub curr_epoch:   Cell<Epoch>,
  pub curr_count:   Cell<u64>,
}

impl Operator for NodeCell {
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
    //self.curr_epoch.get()
    unimplemented!();
  }
}

impl NodeCell {
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

/*pub trait NewDiffOperator2<S> { //: NewDiffOperator<S> {
  //type IoBuf: ?Sized;
  type OpRef: ?Sized + 'static;

  fn _traverse_fwd_new(&mut self, _epoch: u64, _apply: &mut FnMut(&mut Self::OpRef));
  fn _traverse_bwd_new(&mut self, _epoch: u64, _apply: &mut FnMut(&mut Self::OpRef));
  fn _next_new(&self) -> u64 { unimplemented!(); }

  fn _diff_param_sz_new(&mut self) -> usize { 0 }
}

//pub trait NewDiffLoss<S>: NewDiffOperator2<S> where Self::OpRef: Deref<Target=NewDiffOperator2<S, OpRef=Self::OpRef>> {
pub trait NewDiffLoss<S>: NewDiffOperator2<S> where Self::OpRef: NewDiffOpCast<S, OpTarget=Self::OpRef> {
  fn diff_param_sz_new(&mut self) -> usize {
    let epoch = self._next_new();
    let mut grad_sz = 0;
    self._traverse_bwd_new(epoch, &mut |op| {
      unimplemented!();
      //grad_sz += op._diff_param_sz_new();
    });
    grad_sz
  }
}

pub trait NewDiffOpCast<S> {
  type OpTarget: ?Sized;

  fn diff_op(&mut self) -> &mut NewDiffOperator2<S, OpRef=Self::OpTarget>;
}*/

/*pub trait DiffOperatorRma<S, RmaBuf>: NewDiffOperator<S> {
  type RmaCtx;

  fn _rma_load_diff_param(&mut self, offset: usize, param_reader: &mut RmaBuf, ctx: Self::RmaCtx) -> usize { 0 }
  fn _rma_load_nondiff_param(&mut self, offset: usize, param_reader: &mut RmaBuf) -> usize { 0 }
  fn _rma_store_diff_param(&mut self, offset: usize, param_writer: &mut RmaBuf, ctx: Self::RmaCtx) -> usize { 0 }
  fn _rma_store_nondiff_param(&mut self, offset: usize, param_writer: &mut RmaBuf) -> usize { 0 }
  fn _rma_store_grad(&mut self, offset: usize, grad_writer: &mut RmaBuf, ctx: Self::RmaCtx) -> usize { 0 }
}*/

pub enum RwMarker {
  Read,
  Write,
  ReadWrite,
}

pub struct Intermediate<A> {
  pub data:     A,
  pub grad:     A,
  pub r_data:   A,
  pub r_grad:   A,
}

impl Intermediate<RwMarker> {
  pub fn conservative() -> Intermediate<RwMarker> {
    Intermediate{
      data:     RwMarker::ReadWrite,
      grad:     RwMarker::ReadWrite,
      r_data:   RwMarker::ReadWrite,
      r_grad:   RwMarker::ReadWrite,
    }
  }
}

pub struct FnOnceOperator<Args, IoBuf: ?Sized, Buf> {
  out:      Buf,
  _marker:  PhantomData<(fn (Args), fn (IoBuf))>,
}

impl<Args, IoBuf: ?Sized, Buf> FnOnce<Args> for FnOnceOperator<Args, IoBuf, Buf> {
  type Output = Buf;

  extern "rust-call" fn call_once(self, args: Args) -> Buf {
    self.out
  }
}

pub fn test_fn_once_op<S, IoBuf, Buf>(f: FnOnceOperator<(S,), IoBuf, Buf>, a: S) -> Buf {
  f(a)
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParamRef {
  pub node_id:  NodeId,
}

impl ParamRef {
  pub fn new() -> ParamRef {
    ParamRef{node_id: NodeId::new()}
  }
}

#[derive(Clone)]
pub struct ParamSet {
  inner:    HashSet<ParamRef>,
  mask:     HashSet<ParamRef>,
}

impl ParamSet {
  pub fn new() -> ParamSet {
    ParamSet{
      inner:    HashSet::new(),
      mask:     HashSet::new(),
    }
  }

  pub fn contains(&self, param: &ParamRef) -> bool {
    self.inner.contains(param)
  }

  pub fn union(&mut self, other_params: &ParamSet) {
    self.inner.union(&other_params.inner);
    // FIXME(20170115): union the masks too?
  }

  pub fn reset_masks(&mut self) {
    self.mask.clear();
  }

  pub fn mask(&mut self, param: ParamRef) {
    assert!(self.inner.contains(&param));
    self.mask.insert(param);
  }

  pub fn unmasked(&self, param: &ParamRef) -> bool {
    self.inner.contains(param) && !self.mask.contains(param)
  }
}

pub trait ParamAllocator<A> {
  fn allocate_param(&self) -> A;
}

pub struct ParamBuf<A> {
  batch:    Cell<BatchNr>,
  block:    RefCell<Option<Weak<ParamBlock<A>>>>,
  inner:    RefCell<Option<A>>,
}

impl<A> ParamBuf<A> {
  pub fn default() -> ParamBuf<A> {
    ParamBuf{
      batch:    Cell::new(BatchNr::default()),
      block:    RefCell::new(None),
      inner:    RefCell::new(None),
    }
  }

  pub fn set_parent(&self, block: Weak<ParamBlock<A>>) {
    *self.block.borrow_mut() = Some(block);
  }

  pub fn as_ref(&self) -> Ref<A> {
    assert!(self.inner.borrow().is_some());
    Ref::map(self.inner.borrow(), |inner| inner.as_ref().unwrap())
  }

  pub fn as_mut(&self) -> RefMut<A> {
    if self.inner.borrow().is_none() {
      assert!(self.block.borrow().is_some());
      let maybe_block = Weak::upgrade(self.block.borrow().as_ref().unwrap());
      assert!(maybe_block.is_some());
      let block = maybe_block.unwrap();
      let mut inner = self.inner.borrow_mut();
      *inner = Some(block.allocator.allocate_param());
    }
    RefMut::map(self.inner.borrow_mut(), |inner| inner.as_mut().unwrap())
  }
}

pub struct ParamBlock<A> {
  allocator:    Rc<ParamAllocator<A>>,
  //pub node:     NodeStack,
  pub param:    ParamRef,
  pub batch:    Cell<BatchNr>,
  pub val:      ParamBuf<A>,
  pub grad:     ParamBuf<A>,
  pub val2:     ParamBuf<A>,
  pub grad2:    ParamBuf<A>,
  pub r_dir:    ParamBuf<A>,
  pub r_grad:   ParamBuf<A>,
}

impl<A> ParamBlock<A> {
  pub fn new(allocator: Rc<ParamAllocator<A>>) -> Rc<ParamBlock<A>> {
    let block = Rc::new(ParamBlock{
      allocator:    allocator,
      //node:     NodeStack::default(),
      param:    ParamRef::new(),
      batch:    Cell::new(BatchNr::default()),
      val:      ParamBuf::default(),
      grad:     ParamBuf::default(),
      val2:     ParamBuf::default(),
      grad2:    ParamBuf::default(),
      r_dir:    ParamBuf::default(),
      r_grad:   ParamBuf::default(),
    });
    let weak_block = Rc::downgrade(&block);
    block.val.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.grad.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.val2.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.grad2.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.r_dir.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.r_grad.set_parent(weak_block);
    block
  }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Var {
  pub node_id:  NodeId,
}

impl Var {
  pub fn new() -> Var {
    Var{node_id: NodeId::new()}
  }
}

pub trait VarAllocator<A> {
  fn allocate_var(&self) -> A;
}

/*pub trait VarData {
  fn validate(&self, ) -> bool;
}*/

pub struct VarBuf<A> {
  //epoch:    Cell<u64>,
  batch:    Cell<BatchNr>,
  block:    RefCell<Option<Weak<VarBlock<A>>>>,
  inner:    RefCell<Option<A>>,
}

impl<A> VarBuf<A> {
  pub fn default() -> VarBuf<A> {
    VarBuf{
      //epoch:    Cell::new(0),
      batch:    Cell::new(BatchNr::default()),
      block:    RefCell::new(None),
      inner:    RefCell::new(None),
    }
  }

  pub fn set_parent(&self, block: Weak<VarBlock<A>>) {
    *self.block.borrow_mut() = Some(block);
  }

  /*pub fn match_batch(&self, test_batch: BatchNr) -> bool {
    self.batch.get() == test_batch
  }

  pub fn update_batch(&self, new_batch: BatchNr) {
    self.batch.set(new_batch);
  }

  pub fn clear_batch(&self) {
    self.batch.set(BatchNr(0));
  }*/

  pub fn as_ref(&self) -> Ref<A> {
    assert!(self.inner.borrow().is_some());
    Ref::map(self.inner.borrow(), |inner| inner.as_ref().unwrap())
  }

  pub fn as_mut(&self) -> RefMut<A> {
    if self.inner.borrow().is_none() {
      assert!(self.block.borrow().is_some());
      let maybe_block = Weak::upgrade(self.block.borrow().as_ref().unwrap());
      assert!(maybe_block.is_some());
      let block = maybe_block.unwrap();
      let mut inner = self.inner.borrow_mut();
      *inner = Some(block.allocator.allocate_var());
    }
    RefMut::map(self.inner.borrow_mut(), |inner| inner.as_mut().unwrap())
  }
}

pub struct VarBlock<A> {
  allocator:    Rc<VarAllocator<A>>,
  pub var:      Var,
  pub batch:    Cell<BatchNr>,
  pub batch_sz: Cell<usize>,
  pub val:      VarBuf<A>,
  pub grad:     VarBuf<A>,
  pub grad2:    VarBuf<A>,
  pub r_val:    VarBuf<A>,
  pub r_grad:   VarBuf<A>,
}

impl<A> VarBlock<A> {
  pub fn new(allocator: Rc<VarAllocator<A>>) -> Rc<VarBlock<A>> {
    let block = Rc::new(VarBlock{
      allocator:    allocator,
      var:      Var::new(),
      batch:    Cell::new(BatchNr::default()),
      batch_sz: Cell::new(0),
      val:      VarBuf::default(),
      grad:     VarBuf::default(),
      grad2:    VarBuf::default(),
      r_val:    VarBuf::default(),
      r_grad:   VarBuf::default(),
    });
    let weak_block = Rc::downgrade(&block);
    block.val.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.grad.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.grad2.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.r_val.set_parent(weak_block);
    let weak_block = Rc::downgrade(&block);
    block.r_grad.set_parent(weak_block);
    block
  }
}

pub struct DefaultParamAllocator<A, F> where A: 'static, F: 'static + Fn() -> A {
  f:    F,
  _m:   PhantomData<fn () -> A>,
}

impl<A, F> DefaultParamAllocator<A, F> where A: 'static, F: 'static + Fn() -> A {
  pub fn new(f: F) -> Rc<DefaultParamAllocator<A, F>> {
    Rc::new(DefaultParamAllocator{f: f, _m: PhantomData})
  }
}

impl<A, F> ParamAllocator<A> for DefaultParamAllocator<A, F> where F: Fn() -> A {
  fn allocate_param(&self) -> A {
    (self.f)()
  }
}

pub struct DefaultVarAllocator<A, F> where A: 'static, F: 'static + Fn() -> A {
  f:    F,
  _m:   PhantomData<fn () -> A>,
}

impl<A, F> DefaultVarAllocator<A, F> where A: 'static, F: 'static + Fn() -> A {
  pub fn new(f: F) -> Rc<DefaultVarAllocator<A, F>> {
    Rc::new(DefaultVarAllocator{f: f, _m: PhantomData})
  }
}

impl<A, F> VarAllocator<A> for DefaultVarAllocator<A, F> where F: Fn() -> A {
  fn allocate_var(&self) -> A {
    (self.f)()
  }
}

pub trait DiffOperatorData<S> {
  fn _load_batch(&mut self, _samples: &[S]) {}
  fn _cache_batch(&mut self, _keys: &[u64], _cache: &mut Any) { unimplemented!(); }
  fn _reload_cached_batch(&mut self, _keys: &[u64], _cache: &mut Any) { unimplemented!(); }
  /*fn _push_cached_batch(&mut self, _samples: &[S]) {}
  fn _set_cached_batch_weights(&mut self, _weights: &[f32]) {}
  fn _load_cached_batch(&mut self, _idxs: &[usize]) {}*/
}

pub trait DiffOperatorIo<IoBuf: ?Sized> {
  fn _load_diff_param(&mut self, _offset: usize, _param_reader: &mut IoBuf) -> usize { 0 }
  fn _load_nondiff_param(&mut self, _offset: usize, _param_reader: &mut IoBuf) -> usize { 0 }
  fn _store_diff_param(&mut self, _offset: usize, _param_writer: &mut IoBuf) -> usize { 0 }
  fn _store_nondiff_param(&mut self, _offset: usize, _param_writer: &mut IoBuf) -> usize { 0 }
  fn _store_grad(&mut self, _offset: usize, _grad_writer: &mut IoBuf) -> usize { 0 }
  fn _load_direction(&mut self, _offset: usize, _direction_reader: &mut IoBuf) -> usize { 0 }
  fn _store_r_grad(&mut self, _offset: usize, _grad_writer: &mut IoBuf) -> usize { 0 }

  /*fn _load_param(&mut self, _params: &mut ParamSet, _offset: usize, _param_reader: &mut IoBuf) -> usize { 0 }
  fn _store_param(&mut self, _params: &mut ParamSet, _offset: usize, _param_writer: &mut IoBuf) -> usize { 0 }
  fn _store_grad(&mut self, _params: &mut ParamSet, _offset: usize, _grad_writer: &mut IoBuf) -> usize { 0 }
  fn _store_grad2(&mut self, _params: &mut ParamSet, _offset: usize, _grad2_writer: &mut IoBuf) -> usize { 0 }
  fn _load_r_dir(&mut self, _params: &mut ParamSet, _offset: usize, _r_dir_reader: &mut IoBuf) -> usize { 0 }
  fn _store_r_grad(&mut self, _params: &mut ParamSet, _offset: usize, _r_grad_writer: &mut IoBuf) -> usize { 0 }*/
}

/*pub trait DiffOperatorBuf<Src, Sink> {
  fn _src(&self, idx: usize) -> Src;
  fn _sink(&self, idx: usize) -> Sink;
}*/

//pub trait NewDiffOperator<S>: Operator {
pub trait DiffOperator<S, IoBuf: ?Sized>: Operator + DiffOperatorData<S> + DiffOperatorIo<IoBuf> {
  fn _traverse_fwd(&mut self, _epoch: u64 /*Epoch*/, _apply: &mut FnMut(&mut DiffOperator<S, IoBuf>));
  fn _traverse_bwd(&mut self, _epoch: u64 /*Epoch*/, _apply: &mut FnMut(&mut DiffOperator<S, IoBuf>));

  /*fn _fwd_markers(&self) -> (Vec<Intermediate>, Vec<Intermediate>) { unimplemented!(); }
  fn _bwd_markers(&self) -> (Vec<Intermediate>, Vec<Intermediate>) { unimplemented!(); }
  fn _r_fwd_markers(&self) -> (Vec<Intermediate>, Vec<Intermediate>) { unimplemented!(); }
  fn _r_bwd_markers(&self) -> (Vec<Intermediate>, Vec<Intermediate>) { unimplemented!(); }*/

  fn _diff_param_sz(&self) -> usize { 0 }
  fn _nondiff_param_sz(&self) -> usize { 0 }
  fn _diff_params(&self) -> ParamSet { ParamSet::new() }
  fn _nondiff_params(&self) -> ParamSet { ParamSet::new() }
  fn _param_sz(&self, _params: &mut ParamSet) -> usize { 0 }

  fn _save_rng_state(&mut self) {}
  fn _restore_rng_state(&mut self) {}
  fn _load_rng_state(&mut self, offset: usize, state: &[u64]) -> usize { 0 }
  fn _store_rng_state(&mut self, state: &mut Vec<u64>) -> usize { 0 }

  fn _reset_state(&mut self) {}
  fn _reset_accumulators(&mut self) {}

  fn _next_iteration(&mut self) {}
  fn _reset_batch(&mut self) {}
  /*fn _load_batch(&mut self, _samples: &[S]) {}
  fn _push_cached_batch(&mut self, _samples: &[S]) {}
  fn _set_cached_batch_weights(&mut self, _weights: &[f32]) {}
  fn _load_cached_batch(&mut self, _idxs: &[usize]) {}*/

  //fn _reset_state(&mut self) {}
  fn _init_param(&mut self, _rng: &mut Xorshiftplus128Rng) {}
  //fn _init_param(&mut self) {}
  fn _update_nondiff_param(&mut self, _iter_nr: usize) {}
  fn _reset_grad(&mut self) {}
  fn _reset_r_grad(&mut self) {}
  fn _reset_weight(&mut self) {}

  fn _clock(&mut self) { unimplemented!(); }
  fn _forward(&mut self, phase: OpPhase);
  fn _backward(&mut self);
  fn _backward_gauss_newton(&mut self) { self._backward(); }
  fn _backward2(&mut self) { unimplemented!(); }
  fn _r_forward(&mut self) { unimplemented!(); }
  fn _r_forward_gauss_newton(&mut self) { self._r_forward(); }
  fn _r_backward(&mut self) { unimplemented!(); }

  fn _dump_input(&mut self) -> Vec<u8> { unimplemented!(); }
  fn _dump_output(&mut self) -> Vec<u8> { unimplemented!(); }
}

/*pub trait DiffLossRma<S, RmaBuf>: DiffLoss<S> + DiffOperatorRma<S, RmaBuf> {
  fn rma_load_diff_param(&mut self, param_reader: &mut RmaBuf, ctx: Self::RmaCtx) -> usize {
    unimplemented!();
    /*let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._rma_load_diff_param(offset, param_reader);
    });
    offset*/
  }

  fn rma_load_nondiff_param(&mut self, param_reader: &mut RmaBuf, ctx: Self::RmaCtx) -> usize {
    unimplemented!();
    /*let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._rma_load_nondiff_param(offset, param_reader);
    });
    offset*/
  }

  fn rma_store_diff_param(&mut self, param_writer: &mut RmaBuf, ctx: Self::RmaCtx) -> usize {
    unimplemented!();
    /*let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._rma_store_diff_param(offset, param_writer);
    });
    offset*/
  }

  fn rma_store_nondiff_param(&mut self, param_writer: &mut RmaBuf, ctx: Self::RmaCtx) -> usize {
    unimplemented!();
    /*let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._rma_store_nondiff_param(offset, param_writer);
    });
    offset*/
  }

  fn rma_store_grad(&mut self, grad_writer: &mut RmaBuf, ctx: Self::RmaCtx) -> usize {
    unimplemented!();
    /*let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._rma_store_grad(offset, grad_writer);
    });
    offset*/
  }
}*/

/*pub trait NewDiffLoss<S, IoBuf: ?Sized>: DiffOperator<S, IoBuf> {
  fn new_load_diff_param(&mut self, param_reader: &mut IoBuf) -> usize {
    let epoch = self._next();
    let mut offset = 0;
    self._new_traverse_bwd(epoch, &mut |op| {
      offset += op._new_load_diff_param(offset, param_reader);
    });
    offset
  }
}*/

pub trait DiffNLLLoss<S, IoBuf: ?Sized>: DiffLoss<S, IoBuf> {
  fn cache_nll(&mut self);
  fn store_kl_divergence_to_cached(&mut self) -> f32;
}

pub trait DiffLoss<S, IoBuf: ?Sized>: DiffOperator<S, IoBuf> {
  fn reset_loss(&mut self);
  fn store_loss(&mut self);
  fn get_loss(&mut self) -> f32;
  fn set_jacobian_target_with_r_loss(&mut self) { unimplemented!(); }
  fn r_gauss_newton_transform(&mut self) { unimplemented!(); }

  fn _store_accuracy(&mut self) { unimplemented!() }
  fn _get_accuracy(&mut self) -> usize { unimplemented!(); }
  fn _store_pred(&mut self) { unimplemented!(); }
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

  fn load_diff_param(&mut self, param_reader: &mut IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_diff_param(offset, param_reader);
    });
    offset
  }

  fn load_nondiff_param(&mut self, param_reader: &mut IoBuf) -> usize {
    //param_reader.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_nondiff_param(offset, param_reader);
    });
    offset
  }

  fn store_diff_param(&mut self, param_writer: &mut IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_diff_param(offset, param_writer);
    });
    offset
  }

  fn store_nondiff_param(&mut self, param_writer: &mut IoBuf) -> usize {
    //param_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_nondiff_param(offset, param_writer);
    });
    offset
  }

  fn store_grad(&mut self, grad_writer: &mut IoBuf) -> usize {
    //grad_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_grad(offset, grad_writer);
    });
    offset
  }

  fn load_direction(&mut self, dir_reader: &mut IoBuf) -> usize {
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._load_direction(offset, dir_reader);
    });
    offset
  }

  fn store_r_grad(&mut self, grad_writer: &mut IoBuf) -> usize {
    //grad_writer.reset();
    let epoch = self._next();
    let mut offset = 0;
    self._traverse_bwd(epoch, &mut |op| {
      offset += op._store_r_grad(offset, grad_writer);
    });
    offset
  }

  fn next_iteration(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._next_iteration());
  }

  fn reset_batch(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._reset_batch());
  }

  fn load_batch(&mut self, samples: &[S]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._load_batch(samples));
  }

  fn cache_batch(&mut self, keys: &[u64], cache: &mut Any) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._cache_batch(keys, cache));
  }

  fn reload_cached_batch(&mut self, keys: &[u64], cache: &mut Any) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._reload_cached_batch(keys, cache));
  }

  /*fn push_cached_batch(&mut self, samples: &[S]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._push_cached_batch(samples));
  }

  fn set_cached_batch_weights(&mut self, weights: &[f32]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._set_cached_batch_weights(weights));
  }

  fn load_cached_batch(&mut self, idxs: &[usize]) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._load_cached_batch(idxs));
  }*/

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

  fn reset_r_grad(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._reset_r_grad());
  }

  fn reset_weight(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._reset_weight());
  }

  fn forward(&mut self, phase: OpPhase) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._forward(phase));
  }

  fn backward(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._backward());
  }

  fn backward_gauss_newton(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._backward_gauss_newton());
  }

  fn backward2(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._backward2());
  }

  fn r_forward(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._r_forward());
  }

  fn r_forward_gauss_newton(&mut self) {
    let epoch = self._next();
    self._traverse_fwd(epoch, &mut |op| op._r_forward_gauss_newton());
  }

  fn r_backward(&mut self) {
    let epoch = self._next();
    self._traverse_bwd(epoch, &mut |op| op._r_backward());
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

/*pub trait DiffOperator<T> {
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
}*/

pub trait CheckpointFormat {
}

/*pub trait DiffOperatorCheckpoint<Format> where Format: CheckpointFormat {
  fn decode(reader: &mut Read) -> Result<Self, ()> where Self: Sized;
  fn encode(&mut self, writer: &mut Write) -> Result<(), ()>;
}*/
