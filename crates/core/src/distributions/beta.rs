//! Beta distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = α − 1`, `η₂ = β − 1`.
//! Sufficient statistics: `T(x) = (ln x, ln(1−x))`.
//! Log-partition: `A(η) = ln Γ(η₁+1) + ln Γ(η₂+1) − ln Γ(η₁+η₂+2)`.

use nalgebra::DVector;

// Re-use lgamma and digamma from the gamma module.
use super::gamma::{digamma, lgamma};

use super::exp_family::ExponentialFamily;

/// Marker type for the Beta exponential family.
pub(crate) struct BetaDist;

impl ExponentialFamily for BetaDist {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(0.0);
        log_partition(e1, e2)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(0.0);
        expected_suff_stats(e1, e2)
    }

    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// Canonical parameters: `(α, β)`.
fn canonical(eta1: f64, eta2: f64) -> (f64, f64) {
    let alpha = eta1 + 1.0;
    let beta = eta2 + 1.0;
    (alpha, beta)
}

/// `E[T(x)] = (E[ln x], E[ln(1−x)]) = (ψ(α) − ψ(α+β), ψ(β) − ψ(α+β))`.
pub(super) fn expected_suff_stats(eta1: f64, eta2: f64) -> DVector<f64> {
    let (alpha, beta) = canonical(eta1, eta2);
    let psi_sum = digamma(alpha + beta);
    DVector::from_vec(vec![digamma(alpha) - psi_sum, digamma(beta) - psi_sum])
}

/// `A(η) = ln Γ(η₁+1) + ln Γ(η₂+1) − ln Γ(η₁+η₂+2)`.
pub(super) fn log_partition(eta1: f64, eta2: f64) -> f64 {
    let (alpha, beta) = canonical(eta1, eta2);
    lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::exp_family::{cross_entropy as ef_ce, entropy as ef_h};

    /// Reference: Beta(α=2, β=5) → η₁=1, η₂=4.
    const ETA1: f64 = 1.0;
    const ETA2: f64 = 4.0;

    #[test]
    fn canonical_roundtrip() {
        let (alpha, beta) = canonical(ETA1, ETA2);
        assert!((alpha - 2.0).abs() < 1e-12);
        assert!((beta - 5.0).abs() < 1e-12);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(ETA1, ETA2);
        assert_eq!(t.len(), 2);
        let (alpha, beta) = (2.0, 5.0);
        let psi_sum = digamma(alpha + beta);
        let t0 = t.get(0).copied().unwrap_or(f64::NAN);
        let t1 = t.get(1).copied().unwrap_or(f64::NAN);
        assert!((t0 - (digamma(alpha) - psi_sum)).abs() < 1e-10);
        assert!((t1 - (digamma(beta) - psi_sum)).abs() < 1e-10);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let eta = DVector::from_vec(vec![ETA1, ETA2]);
        let ce = ef_ce::<BetaDist>(&eta, &eta);
        let h = ef_h::<BetaDist>(&eta);
        assert!((ce + h).abs() < 1e-10);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1, ETA2);
        let expected = lgamma(2.0) + lgamma(5.0) - lgamma(7.0);
        assert!((a - expected).abs() < 1e-10);
    }
}
