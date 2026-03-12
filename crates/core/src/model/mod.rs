//! Core model components — graph, inference, scoring, temporal dynamics.
//!
//! This module groups the runtime model: the patient graph structure,
//! the inference engine (Jacobi fixed-point), the scoring function,
//! temporal relaxation, and shared constants.

pub mod constants;
pub mod graph;
pub mod inference;
pub mod score;
pub mod temporal;
