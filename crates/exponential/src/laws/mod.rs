//! Concrete exponential family distributions.
//!
//! Each "law" implements [`ExponentialFamily`](crate::distribution::ExponentialFamily)
//! and provides conversions between its standard parameterization and the
//! natural (canonical) exponential family form:
//!
//! - **From standard → natural**: `to_natural` returns `η ∈ ℝᵏ`
//! - **From natural → standard**: `from_natural` recovers the usual parameters
//!
//! # Available laws
//!
//! | Law | Standard params | Natural params `η` | Sufficient stat `T(x)` |
//! |-----|----------------|-------------------|----------------------|
//! | [`Normal1D`] | `(μ, σ²)` | `(μ/σ², −1/(2σ²))` | `(x, x²)` |

/// 1-D Normal (Gaussian) distribution.
pub mod gaussian;
