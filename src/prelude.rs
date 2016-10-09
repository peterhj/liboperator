pub use super::{
  DiffOperator, DiffOperatorInput, DiffOperatorOutput, DiffOperatorIo, OpCapability, OpPhase, Regularization,
};
pub use data::Shape::*;
pub use data::{
  ClassLoss, RegressLoss,
  SampleDatum, SampleLabel, SampleLossWeight,
};
pub use opt::{
  OptWorker, OptStats, StepSize, AdaptiveStepSizeSchedule,
};
