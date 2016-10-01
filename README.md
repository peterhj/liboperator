# liboperator

This is a Rust library for implementing differentiable operators used in
optimization. This library contains traits for generic operators, as well as
some implementations of common optimization algorithms, e.g. SGD.

## Concepts

The core trait `DiffOperator` represents a differentiable operator and provides
an interface for the following tasks:

- loading external data (input, class labels, regression targets, etc.)
- loading/storing a differentiable parameter set to a buffer
- loading/storing a non-differentiable parameter set to a buffer
- storing the gradient to a buffer
- storing the loss or objective
- forward computation of the operator's underlying function
- backward propagation to compute gradients
- R-forward and R-backward propagation to compute Jacobian- and Hessian-vector
  products

## I/O

A significant responsibility of `DiffOperator` is to expose a reusable interface
of I/O methods for loading and storing things of interest like data, parameters,
gradients, the loss, and perhaps other objects.

For data, the `DiffOperatorInput` trait allows specialized loading semantics for
different types of data. For example, a classification loss operator will want
to implement `DiffOperatorInput` with a sample data type constrained to those
with a class label (i.e. implementing `SampleClass`).

For parameters and gradients, the approach this library takes is to provide
`ReadBuffer` and `WriteBuffer` traits for respectively loading and storing
typed slices, as well as `ReadAccumulateBuffer` and `AccumulateBuffer` for
combining a load/store with a linear combination.
