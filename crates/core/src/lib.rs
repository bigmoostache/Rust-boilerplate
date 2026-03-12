//! Core library for the Baillissime virtual patient model.
//!
//! This crate implements a probabilistic graphical model (MRF) where
//! each node carries a distribution from an exponential family.
//! Inference produces a coherent Bayesian update of the patient state.

pub mod distributions;
pub mod graph;
pub mod inference;
pub mod schema;
pub mod score;
pub mod temporal;
