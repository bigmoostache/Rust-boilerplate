//! Gaussian (Normal) distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = μ/σ²`, `η₂ = −1/(2σ²)`.
//! Sufficient statistics: `T(x) = (x, x²)`.
//! Log-partition: `A(η) = −η₁²/(4η₂) + ½ ln(−π/η₂)`.

use nalgebra::{DMatrix, DVector};

use crate::constants::NATURAL_PARAM_EPS;

use super::ExponentialFamily;

/// Marker type for the Gaussian exponential family.
pub(crate) struct Gaussian;

impl ExponentialFamily for Gaussian {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-0.5);
        log_partition(e1, e2)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-0.5);
        expected_suff_stats(e1, e2)
    }

    /// `F = Cov[(x, x²)]`:
    /// - `Var[x] = σ²`
    /// - `Cov[x, x²] = 2μσ²`
    /// - `Var[x²] = 2σ⁴ + 4μ²σ²`
    fn fisher_information(eta: &DVector<f64>) -> DMatrix<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-0.5);
        let (mu, sigma2) = canonical(e1, e2);
        let cov_x_x2 = 2.0 * mu * sigma2;
        let var_x2 = (2.0f64).mul_add(sigma2 * sigma2, 4.0 * mu * mu * sigma2);
        DMatrix::from_row_slice(2, 2, &[sigma2, cov_x_x2, cov_x_x2, var_x2])
    }

    /// Returns `0` because we use the convention `h(x) = 1`.
    ///
    /// The standard Gaussian base measure `h(x) = (2π)^{−½}` is
    /// absorbed into `A(η)` (which includes `½ ln(2πσ²)`).
    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }

    /// Clamp `η₂ < −ε` (ensures `σ² > 0`).
    fn project(eta: &mut DVector<f64>) {
        if let Some(e2) = eta.get_mut(1)
            && *e2 >= -NATURAL_PARAM_EPS
        {
            *e2 = -NATURAL_PARAM_EPS;
        }
    }
}

/// Canonical parameters `(μ, σ²)` recovered from natural form.
fn canonical(eta1: f64, eta2: f64) -> (f64, f64) {
    let sigma2 = -1.0 / (2.0 * eta2);
    let mu = eta1 * sigma2;
    (mu, sigma2)
}

/// `E[T(x)] = (E[x], E[x²]) = (μ, μ² + σ²)`.
pub(super) fn expected_suff_stats(eta1: f64, eta2: f64) -> DVector<f64> {
    let (mu, sigma2) = canonical(eta1, eta2);
    DVector::from_vec(vec![mu, mu.mul_add(mu, sigma2)])
}

/// `A(η) = −η₁²/(4η₂) + ½ ln(π/(−η₂))`.
///
/// This uses the convention `h(x) = 1`, absorbing the `(2π)^{−½}`
/// normalizer into `A(η)`.  With `−η₂ = 1/(2σ²)`, the second term
/// equals `½ ln(2πσ²)`.
pub(super) fn log_partition(eta1: f64, eta2: f64) -> f64 {
    let quadratic = -eta1 * eta1 / (4.0 * eta2);
    0.5f64.mul_add((-std::f64::consts::PI / eta2).ln(), quadratic)
}

#[cfg(test)]
mod tests {
    use super::super::{ef_cross_entropy as ef_ce, ef_entropy as ef_h};
    use super::*;

    /// Reference: N(3, 4) → μ=3, σ²=4, η₁ = 0.75, η₂ = −0.125.
    const ETA1: f64 = 0.75;
    const ETA2: f64 = -0.125;

    #[test]
    fn canonical_roundtrip() {
        let (mu, sigma2) = canonical(ETA1, ETA2);
        assert!((mu - 3.0).abs() < 1e-12);
        assert!((sigma2 - 4.0).abs() < 1e-12);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(ETA1, ETA2);
        assert_eq!(t.len(), 2);
        let ex = t.get(0).copied().unwrap_or(f64::NAN);
        let ex2 = t.get(1).copied().unwrap_or(f64::NAN);
        assert!((ex - 3.0).abs() < 1e-12);
        assert!((ex2 - 13.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_value() {
        // H(N(μ,σ²)) = ½ ln(2πeσ²) = ½ ln(2πe·4)
        let eta = DVector::from_vec(vec![ETA1, ETA2]);
        let h = ef_h::<Gaussian>(&eta);
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * 4.0).ln();
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let eta = DVector::from_vec(vec![ETA1, ETA2]);
        let ce = ef_ce::<Gaussian>(&eta, &eta);
        let h = ef_h::<Gaussian>(&eta);
        assert!((ce + h).abs() < 1e-12);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1, ETA2);
        // A(η) = −η₁²/(4η₂) + ½ ln(−π/η₂)
        // = −0.5625 / (−0.5) + ½ ln(π/0.125)
        // = 1.125 + ½ ln(8π)
        let expected = 0.5f64.mul_add((8.0 * std::f64::consts::PI).ln(), 1.125);
        assert!((a - expected).abs() < 1e-12);
    }
}
