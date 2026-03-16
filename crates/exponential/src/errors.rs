//! Error types for the exponential family crate.

use core::fmt;

/// Errors that can occur in exponential family operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExponentialError {
    /// The natural parameter dimension does not match the family's expected dimension.
    NaturalParameterDimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },

    /// The sufficient statistic dimension does not match the family's expected dimension.
    SufficientStatisticDimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },

    /// The pseudo-observation count must be strictly positive.
    NonPositivePseudoCount(f64),

    /// The log-partition value is not finite (NaN or infinity).
    NonFiniteLogPartition(f64),

    /// The base measure value is not finite or is negative.
    InvalidBaseMeasure(f64),

    /// The observation count exceeds the maximum representable as `u32`.
    ObservationCountOverflow(usize),
}

impl fmt::Display for ExponentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaturalParameterDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "natural parameter dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::SufficientStatisticDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "sufficient statistic dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::NonPositivePseudoCount(nu) => {
                write!(f, "pseudo-observation count must be > 0, got {nu}")
            }
            Self::NonFiniteLogPartition(val) => {
                write!(f, "log-partition value is not finite: {val}")
            }
            Self::InvalidBaseMeasure(val) => {
                write!(f, "base measure value is not finite or is negative: {val}")
            }
            Self::ObservationCountOverflow(n) => {
                write!(
                    f,
                    "observation count {n} exceeds maximum representable as u32"
                )
            }
        }
    }
}
