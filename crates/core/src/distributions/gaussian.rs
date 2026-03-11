//! Gaussian (Normal) distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = μ/σ²`, `η₂ = −1/(2σ²)`.
//! Sufficient statistics: `T(x) = (x, x²)`.
//! Log-partition: `A(η) = −η₁²/(4η₂) + ½ ln(−π/η₂)`.

use nalgebra::DVector;

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

/// `H = ½ ln(2πeσ²)`.
pub(super) fn entropy(eta1: f64, eta2: f64) -> f64 {
    let (_, sigma2) = canonical(eta1, eta2);
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * sigma2).ln()
}

/// `E_self[ln p_other(x)]` where both are Gaussian.
///
/// `= η_other · E_self[T(x)] − A(η_other)`
/// `= η₁' · E[x] + η₂' · E[x²] − A(η')`
pub(super) fn cross_entropy(me_eta1: f64, me_eta2: f64, other_eta1: f64, other_eta2: f64) -> f64 {
    let (mu, sigma2) = canonical(me_eta1, me_eta2);
    let e_x = mu;
    let e_x2 = mu.mul_add(mu, sigma2);
    let a_other = log_partition(other_eta1, other_eta2);
    other_eta1.mul_add(e_x, other_eta2 * e_x2) - a_other
}

/// `A(η) = −η₁²/(4η₂) + ½ ln(−π/η₂)`.
pub(super) fn log_partition(eta1: f64, eta2: f64) -> f64 {
    let quadratic = -eta1 * eta1 / (4.0 * eta2);
    0.5f64.mul_add((-std::f64::consts::PI / eta2).ln(), quadratic)
}

#[cfg(test)]
mod tests {
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
        assert!((t[0] - 3.0).abs() < 1e-12); // E[x] = μ
        assert!((t[1] - 13.0).abs() < 1e-12); // E[x²] = μ² + σ² = 9 + 4
    }

    #[test]
    fn entropy_value() {
        // H(N(μ,σ²)) = ½ ln(2πeσ²) = ½ ln(2πe·4)
        let h = entropy(ETA1, ETA2);
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * 4.0).ln();
        assert!((h - expected).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let ce = cross_entropy(ETA1, ETA2, ETA1, ETA2);
        let h = entropy(ETA1, ETA2);
        assert!((ce + h).abs() < 1e-12);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1, ETA2);
        // A(η) = −η₁²/(4η₂) + ½ ln(−π/η₂)
        // = −0.5625 / (−0.5) + ½ ln(π/0.125)
        // = 1.125 + ½ ln(8π)
        let expected = 1.125 + 0.5 * (8.0 * std::f64::consts::PI).ln();
        assert!((a - expected).abs() < 1e-12);
    }
}
