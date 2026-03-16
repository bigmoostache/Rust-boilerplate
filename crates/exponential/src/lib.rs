//! Generic exponential family distributions with conjugate Bayesian inference.
//!
//! # Overview
//!
//! An exponential family distribution has the canonical form:
//!
//! ```text
//! p(x | η) = h(x) · exp(ηᵀ T(x) − A(η))
//! ```
//!
//! where:
//! - `η ∈ ℝᵏ` is the **natural parameter**
//! - `T(x)` is the **sufficient statistic** (maps data → ℝᵏ)
//! - `A(η)` is the **log-partition function** (ensures normalization)
//! - `h(x)` is the **base measure**
//!
//! This crate provides:
//! - [`ExponentialFamily`]: trait defining any member of the exponential family
//! - [`ConjugatePrior`]: the universal conjugate prior `p(η | λ, ν) ∝ exp(ηᵀλ − ν A(η))`
//! - Bayesian update: `λₙ = λ₀ + Σ T(xᵢ)`, `νₙ = ν₀ + n`

/// The [`ExponentialFamily`](distribution::ExponentialFamily) trait.
pub mod distribution;
/// Error types for exponential family operations.
pub mod errors;
/// The [`ConjugatePrior`](inference::ConjugatePrior) struct.
pub mod inference;
/// Concrete exponential family distributions (Normal, etc.).
pub mod laws;
