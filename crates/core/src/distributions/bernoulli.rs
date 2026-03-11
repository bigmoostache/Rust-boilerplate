//! Bernoulli distribution — closed-form computations.
//!
//! Natural parameter: `η₁ = logit(p) = ln(p/(1−p))`.
//! Sufficient statistic: `T(x) = x`.
//! Log-partition: `A(η) = ln(1 + exp(η₁))`.

use nalgebra::DVector;

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

/// `H(Bernoulli(p)) = −p ln p − (1−p) ln(1−p)`.
pub(super) fn entropy(eta1: f64) -> f64 {
    let p = canonical(eta1);
    if !(1e-15..=1.0 - 1e-15).contains(&p) {
        return 0.0; // Degenerate
    }
    (-p).mul_add(p.ln(), -(1.0 - p) * (1.0 - p).ln())
}

/// `E_self[ln p_other(x)]` where both are Bernoulli.
///
/// `= η_other · E_self[T(x)] − A(η_other)`
/// `= η₁' · p − ln(1 + exp(η₁'))`
pub(super) fn cross_entropy(me_eta1: f64, other_eta1: f64) -> f64 {
    let p = canonical(me_eta1);
    let a_other = log_partition(other_eta1);
    other_eta1.mul_add(p, -a_other)
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
        let h = entropy(ETA1);
        let expected = (-0.7_f64).mul_add(0.7_f64.ln(), -0.3 * 0.3_f64.ln());
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let ce = cross_entropy(ETA1, ETA1);
        let h = entropy(ETA1);
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
        assert!(entropy(-30.0).abs() < 1e-10);

        // p ≈ 1 → η → +∞
        let p_high = canonical(30.0);
        assert!((p_high - 1.0).abs() < 1e-12);
        assert!(entropy(30.0).abs() < 1e-10);
    }
}
