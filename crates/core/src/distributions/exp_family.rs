//! Exponential family trait — generic formulas for entropy, cross-entropy.
//!
//! An exponential family density has the form:
//!
//! ```text
//! p(x; η) = h(x) · exp(η · T(x) − A(η))
//! ```
//!
//! where `η` are natural parameters, `T(x)` the sufficient statistics,
//! `A(η)` the log-partition function, and `h(x)` the base measure.
//!
//! Given `A` and `E[ln h(x)]`, entropy and cross-entropy follow
//! automatically:
//!
//! ```text
//! H(η)                   = −E[ln h] − η · E[T] + A(η)
//! E_me[ln p_other(x)]    =  E[ln h] + η_other · E_me[T] − A(η_other)
//! ```
//!
//! Each family only needs to provide `log_partition`, `expected_suff_stats`,
//! and (for Poisson/Dirichlet) `expected_log_base_measure`.  Everything
//! else is derived.

use nalgebra::DVector;

/// Trait for exponential family distributions.
///
/// Implementors provide the three family-specific ingredients;
/// generic functions [`entropy`] and [`cross_entropy`] derive the rest.
pub(crate) trait ExponentialFamily {
    /// Log-partition function `A(η)`.
    fn log_partition(eta: &DVector<f64>) -> f64;

    /// Expected sufficient statistics `E_η[T(x)] = ∇A(η)`.
    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64>;

    /// Expected log-base-measure `E_η[ln h(x)]`.
    ///
    /// Defaults to `0` — correct for families where `h(x) = 1` or
    /// where `h(x)` is absorbed into `A(η)` (Gaussian, Gamma, Beta,
    /// Bernoulli, Categorical).
    ///
    /// Override for Poisson (`−E[ln(x!)]`) and Dirichlet.
    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// Differential entropy `H(η) = −E_η[ln p_η(x)]`.
///
/// ```text
/// H = −E[ln h(x)] − η · E[T(x)] + A(η)
/// ```
pub(crate) fn entropy<F: ExponentialFamily>(eta: &DVector<f64>) -> f64 {
    -F::expected_log_base_measure(eta) - eta.dot(&F::expected_suff_stats(eta))
        + F::log_partition(eta)
}

/// Cross-entropy `E_me[ln p_other(x)]`.
///
/// ```text
/// E_me[ln p_other] = E_me[ln h(x)] + η_other · E_me[T(x)] − A(η_other)
/// ```
pub(crate) fn cross_entropy<F: ExponentialFamily>(
    me: &DVector<f64>,
    other: &DVector<f64>,
) -> f64 {
    F::expected_log_base_measure(me) + other.dot(&F::expected_suff_stats(me))
        - F::log_partition(other)
}
