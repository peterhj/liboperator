# liboperator

This is a Rust library for implementing differentiable operators used in
optimization. This library contains traits for generic operators, as well as
some implementations of common optimization algorithms, e.g. SGD.

## Concepts

The core trait `DiffOperator` represents a differentiable operator and provides
an interface for the following tasks:

- loading external data (e.g. input, class labels, regression targets, etc.)
- loading/storing the differentiable part of the parameter to a buffer
- loading/storing the non-differentiable part of the parameter to a buffer
  (e.g. empirical batch statistics like mean and variance)
- storing the gradient to a buffer
- storing the loss or objective value
- random initialization of the parameter
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
to implement `DiffOperatorInput` with a sample data type parameter constrained
to having a class label (i.e. implementing `SampleClass`).

For parameters and gradients, the approach this library takes is to provide
`ReadBuffer` and `WriteBuffer` traits for respectively loading and storing
typed slices, as well as `ReadAccumulateBuffer` and `AccumulateBuffer` for
combining a load/store with a linear combination.

## Optimization

Iterative optimization methods implement the `OptWorker` trait, mainly the
`step` method. This trait also makes a distinction between "local" and "global"
parameters; this is useful primarily for distributed algorithms.
