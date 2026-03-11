//! YAML schema types and strict validation.
//!
//! This module defines serde-compatible types for loading a patient
//! graph from YAML.  Input uses **canonical parameters** (human-readable)
//! that are converted to natural parameters during validation.
//!
//! Validation is strict and exhaustive — all errors are collected
//! before being reported.  No unknown fields are accepted.

pub mod diag;
mod observation_compat;
pub mod output;
pub mod raw;
#[cfg(test)]
mod tests;
pub mod validate;
