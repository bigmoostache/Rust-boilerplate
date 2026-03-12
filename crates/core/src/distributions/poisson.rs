//! Poisson distribution — closed-form computations.
//!
//! Natural parameter: `η₁ = ln λ`.
//! Sufficient statistic: `T(x) = x`.
//! Log-partition: `A(η) = exp(η₁) = λ`.
//!
//! Note: the Poisson entropy involves an infinite series and has no
//! elementary closed form. We use the approximation
//! `H ≈ ½ ln(2πeλ) − 1/(12λ) − 1/(24λ²) − …` for `λ ≥ 1`,
//! and exact summation for small `λ`.

use nalgebra::{DMatrix, DVector};

use super::ExponentialFamily;

/// Marker type for the Poisson exponential family.
pub(crate) struct PoissonDist;

impl ExponentialFamily for PoissonDist {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        log_partition(e1)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        expected_suff_stats(e1)
    }

    /// `Var[x] = λ = e^{η₁}`.
    fn fisher_information(eta: &DVector<f64>) -> DMatrix<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        DMatrix::from_row_slice(1, 1, &[canonical(e1)])
    }

    /// `E_η[ln h(x)] = −E_η[ln(x!)]`.
    ///
    /// The Poisson base measure is `h(x) = 1/x!`.
    fn expected_log_base_measure(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        -expected_log_factorial(e1)
    }
}

/// Canonical parameter: `λ = exp(η₁)`.
fn canonical(eta1: f64) -> f64 {
    eta1.exp()
}

/// `E[T(x)] = E[x] = λ`.
pub(super) fn expected_suff_stats(eta1: f64) -> DVector<f64> {
    DVector::from_vec(vec![canonical(eta1)])
}

/// `E_λ[ln(x!)]` — needed for the base measure in Poisson.
fn expected_log_factorial(eta1: f64) -> f64 {
    let lambda = canonical(eta1);
    if lambda < 1e-15 {
        return 0.0;
    }
    if lambda >= 50.0 {
        // Stirling-based approximation
        return 0.5f64.mul_add(
            (2.0 * std::f64::consts::PI * lambda).ln(),
            lambda.mul_add(lambda.ln(), -lambda),
        ) + 1.0 / (12.0 * lambda);
    }
    // Direct summation
    let neg_lambda = -lambda;
    let ln_lambda = lambda.ln();
    let mut result = 0.0_f64;
    let mut log_p = neg_lambda;
    let mut log_factorial = 0.0_f64; // ln(0!) = 0
    for k in 0..=1000_u32 {
        let p = log_p.exp();
        if p > 1e-15 {
            result += p * log_factorial;
        }
        if p < 1e-15 && f64::from(k) > lambda {
            break;
        }
        let Some(k1) = k.checked_add(1) else { break };
        let next = f64::from(k1);
        log_p += ln_lambda - next.ln();
        log_factorial += next.ln();
    }
    result
}

/// `A(η) = exp(η₁) = λ`.
pub(super) fn log_partition(eta1: f64) -> f64 {
    canonical(eta1)
}

#[cfg(test)]
mod tests {
    use super::super::{ef_cross_entropy as ef_ce, ef_entropy as ef_h};
    use super::*;

    /// Reference: Poisson(λ=5) → η₁ = ln 5.
    const ETA1: f64 = 1.609_437_912_434_100_4; // ln(5)

    #[test]
    fn canonical_value() {
        let lambda = canonical(ETA1);
        assert!((lambda - 5.0).abs() < 1e-12);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(ETA1);
        let val = t.get(0).copied().unwrap_or(f64::NAN);
        assert!((val - 5.0).abs() < 1e-12);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1);
        assert!((a - 5.0).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        // E_p[ln p(x)] = −H(p) when self == other
        let eta = DVector::from_vec(vec![ETA1]);
        let ce = ef_ce::<PoissonDist>(&eta, &eta);
        let h = ef_h::<PoissonDist>(&eta);
        assert!(
            (ce + h).abs() < 1e-6,
            "ce={ce}, -h={}, diff={}",
            -h,
            (ce + h).abs()
        );
    }

    #[test]
    fn entropy_positive() {
        // Poisson entropy should be positive for λ > 0
        let eta = DVector::from_vec(vec![ETA1]);
        let h = ef_h::<PoissonDist>(&eta);
        assert!(h > 0.0, "entropy should be positive, got {h}");
    }
}
