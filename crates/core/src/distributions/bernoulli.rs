//! Bernoulli distribution — closed-form computations.
//!
//! Natural parameter: `η₁ = logit(p) = ln(p/(1−p))`.
//! Sufficient statistic: `T(x) = x`.
//! Log-partition: `A(η) = ln(1 + exp(η₁))`.

use nalgebra::DVector;

use super::exp_family::ExponentialFamily;

/// Marker type for the Bernoulli exponential family.
pub(crate) struct Bernoulli;

impl ExponentialFamily for Bernoulli {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        log_partition(e1)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        expected_suff_stats(e1)
    }

    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// Canonical parameter: `p = sigmoid(η₁) = 1/(1 + exp(−η₁))`.
fn canonical(eta1: f64) -> f64 {
    // Numerically stable sigmoid
    if eta1 >= 0.0 {
        let e = (-eta1).exp();
        1.0 / (1.0 + e)
    } else {
        let e = eta1.exp();
        e / (1.0 + e)
    }
}

/// `E[T(x)] = E[x] = p`.
pub(super) fn expected_suff_stats(eta1: f64) -> DVector<f64> {
    DVector::from_vec(vec![canonical(eta1)])
}

/// `A(η) = ln(1 + exp(η₁))` (softplus).
pub(super) fn log_partition(eta1: f64) -> f64 {
    // Numerically stable softplus: ln(1 + exp(η))
    if eta1 >= 0.0 {
        eta1 + (-eta1).exp().ln_1p()
    } else {
        eta1.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::exp_family::{cross_entropy as ef_ce, entropy as ef_h};

    /// Reference: Bernoulli(p=0.7) → η₁ = logit(0.7) = ln(7/3).
    const ETA1: f64 = 0.847_297_860_387_203_8; // ln(7/3)

    #[test]
    fn canonical_value() {
        let p = canonical(ETA1);
        assert!((p - 0.7).abs() < 1e-12);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(ETA1);
        let p = t.get(0).copied().unwrap_or(f64::NAN);
        assert!((p - 0.7).abs() < 1e-12);
    }

    #[test]
    fn entropy_value() {
        let eta = DVector::from_vec(vec![ETA1]);
        let h = ef_h::<Bernoulli>(&eta);
        let expected = (-0.7_f64).mul_add(0.7_f64.ln(), -0.3 * 0.3_f64.ln());
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let eta = DVector::from_vec(vec![ETA1]);
        let ce = ef_ce::<Bernoulli>(&eta, &eta);
        let h = ef_h::<Bernoulli>(&eta);
        assert!((ce + h).abs() < 1e-12);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1);
        // A(η) = ln(1 + exp(η)) = ln(1 + 7/3) = ln(10/3)
        let expected = (10.0_f64 / 3.0).ln();
        assert!((a - expected).abs() < 1e-12);
    }

    #[test]
    fn extreme_values_converge() {
        // p ≈ 0 → η → −∞
        let p_low = canonical(-30.0);
        assert!(p_low < 1e-12);
        let eta_low = DVector::from_vec(vec![-30.0]);
        assert!(ef_h::<Bernoulli>(&eta_low).abs() < 1e-10);

        // p ≈ 1 → η → +∞
        let p_high = canonical(30.0);
        assert!((p_high - 1.0).abs() < 1e-12);
        let eta_high = DVector::from_vec(vec![30.0]);
        assert!(ef_h::<Bernoulli>(&eta_high).abs() < 1e-10);
    }
}
