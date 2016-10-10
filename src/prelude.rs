pub use super::{
  DiffOperator, DiffOperatorInput, DiffOperatorOutput, DiffOperatorIo, OpCapability, OpPhase, Regularization,
};
pub use data::{
  ClassLoss, RegressLoss, Shape,
  SampleDatum, SampleLabel, SampleLossWeight,
};
pub use data::Shape::*;
pub use opt::{
  OptWorker, OptStats, StepSize, GradientMomentum, AdaptiveStepSizeSchedule,
};
