//! Core library for the Baillissime virtual patient model.
//!
//! This crate implements a probabilistic graphical model (MRF) where
//! each node carries a distribution from an exponential family.
//! Inference produces a coherent Bayesian update of the patient state.

pub mod calibration;
pub mod distributions;
pub mod model;
pub mod schema;
