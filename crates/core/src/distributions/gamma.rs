//! Gamma distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = α − 1`, `η₂ = −β`.
//! Sufficient statistics: `T(x) = (ln x, x)`.
//! Log-partition: `A(η) = ln Γ(η₁ + 1) − (η₁ + 1) ln(−η₂)`.

use nalgebra::{DMatrix, DVector};

use crate::constants::{DIGAMMA_ASYM, TRIGAMMA_ASYM};

use super::ExponentialFamily;

/// Marker type for the Gamma exponential family.
pub(crate) struct GammaDist;

impl ExponentialFamily for GammaDist {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-1.0);
        log_partition(e1, e2)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-1.0);
        expected_suff_stats(e1, e2)
    }

    /// `F = Cov[(ln x, x)]`:
    /// - `Var[ln x] = ψ'(α)`
    /// - `Cov[ln x, x] = 1/β`
    /// - `Var[x] = α/β²`
    fn fisher_information(eta: &DVector<f64>) -> DMatrix<f64> {
        let e1 = eta.get(0).copied().unwrap_or(0.0);
        let e2 = eta.get(1).copied().unwrap_or(-1.0);
        let (alpha, beta) = canonical(e1, e2);
        let var_lnx = trigamma(alpha);
        let cov_lnx_x = 1.0 / beta;
        let var_x = alpha / (beta * beta);
        DMatrix::from_row_slice(2, 2, &[var_lnx, cov_lnx_x, cov_lnx_x, var_x])
    }

    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// Canonical parameters: `(α, β)`.
fn canonical(eta1: f64, eta2: f64) -> (f64, f64) {
    let alpha = eta1 + 1.0;
    let beta = -eta2;
    (alpha, beta)
}

/// Digamma function `ψ(x) = d/dx ln Γ(x)`.
///
/// Uses the asymptotic series with argument reduction.
///
/// ```text
/// ψ(z) = ln(z) − 1/(2z) − Σ_k B_{2k} / (2k · z^{2k})
/// ```
pub(crate) fn digamma(x: f64) -> f64 {
    let mut result = 0.0;
    let mut z = x;
    // Shift z up until z >= 8 for good asymptotic accuracy.
    for _ in 0..8_u32 {
        if z >= 8.0 {
            break;
        }
        result -= 1.0 / z;
        z += 1.0;
    }
    // Asymptotic expansion: ψ(z) = ln(z) − 1/(2z) − Σ c_k / z^{2k}
    // where c_k = B_{2k} / (2k) = DIGAMMA_ASYM[k-1].
    // Evaluate via Horner in z⁻².
    let iz2 = 1.0 / (z * z);
    let mut poly = 0.0_f64;
    for &c in DIGAMMA_ASYM.iter().rev() {
        poly = poly.mul_add(iz2, c);
    }
    result += z.ln() - 0.5 / z - iz2 * poly;
    result
}

/// Trigamma function `ψ'(x) = d²/dx² ln Γ(x)`.
///
/// Uses the asymptotic series with argument reduction.
///
/// ```text
/// ψ'(z) = 1/z + 1/(2z²) + Σ_k B_{2k} / z^{2k+1}
/// ```
pub(crate) fn trigamma(x: f64) -> f64 {
    let mut result = 0.0;
    let mut z = x;
    // Recurrence: ψ'(x) = ψ'(x+1) + 1/x²
    for _ in 0..8_u32 {
        if z >= 8.0 {
            break;
        }
        result += 1.0 / (z * z);
        z += 1.0;
    }
    // Asymptotic expansion: ψ'(z) = 1/z + 1/(2z²) + Σ B_{2k} / z^{2k+1}
    // = 1/z + 1/(2z²) + (1/z³) · Σ B_{2k} · z^{−2(k−1)}
    // Evaluate the polynomial Σ B_{2k} · (1/z²)^{k-1} via Horner in z⁻².
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    let mut poly = 0.0_f64;
    for &c in TRIGAMMA_ASYM.iter().rev() {
        poly = poly.mul_add(iz2, c);
    }
    result += (iz2 * iz).mul_add(poly, 0.5f64.mul_add(iz2, iz));
    result
}

/// `E[T(x)] = (E[ln x], E[x]) = (ψ(α) − ln β, α/β)`.
pub(super) fn expected_suff_stats(eta1: f64, eta2: f64) -> DVector<f64> {
    let (alpha, beta) = canonical(eta1, eta2);
    DVector::from_vec(vec![digamma(alpha) - beta.ln(), alpha / beta])
}

/// `A(η) = ln Γ(η₁ + 1) − (η₁ + 1) ln(−η₂)`.
pub(super) fn log_partition(eta1: f64, eta2: f64) -> f64 {
    let alpha = eta1 + 1.0;
    let beta = -eta2;
    lgamma(alpha) - alpha * beta.ln()
}

/// `ln Γ(x)` — log-gamma function via Stirling + Lanczos.
pub(crate) fn lgamma(x: f64) -> f64 {
    // Use the standard library's implementation via ln_gamma.
    // Rust's f64 doesn't have ln_gamma in std, so we use a direct
    // Lanczos approximation (g=7, n=9).
    let coeffs: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = coeffs[0];
    for (i, &c) in coeffs.iter().enumerate().skip(1) {
        let i_f64 = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
        sum += c / (z + i_f64);
    }
    let t = z + 7.5;
    0.5f64.mul_add(
        (2.0 * std::f64::consts::PI).ln(),
        (z + 0.5).mul_add(t.ln(), -t),
    ) + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::super::{ef_cross_entropy as ef_ce, ef_entropy as ef_h};
    use super::*;

    /// Reference: Gamma(α=3, β=2) → η₁=2, η₂=−2.
    const ETA1: f64 = 2.0;
    const ETA2: f64 = -2.0;

    #[test]
    fn canonical_roundtrip() {
        let (alpha, beta) = canonical(ETA1, ETA2);
        assert!((alpha - 3.0).abs() < 1e-12);
        assert!((beta - 2.0).abs() < 1e-12);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(ETA1, ETA2);
        assert_eq!(t.len(), 2);
        let psi3 = digamma(3.0);
        let t0 = t.get(0).copied().unwrap_or(f64::NAN);
        let t1 = t.get(1).copied().unwrap_or(f64::NAN);
        assert!((t0 - (psi3 - 2.0_f64.ln())).abs() < 1e-10);
        // E[x] = α/β = 1.5
        assert!((t1 - 3.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let eta = DVector::from_vec(vec![ETA1, ETA2]);
        let ce = ef_ce::<GammaDist>(&eta, &eta);
        let h = ef_h::<GammaDist>(&eta);
        assert!((ce + h).abs() < 1e-10);
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = −γ ≈ −0.5772156649
        assert!((digamma(1.0) + 0.577_215_664_901_532_9).abs() < 1e-10);
        // ψ(2) = 1 − γ
        assert!((digamma(2.0) - (1.0 - 0.577_215_664_901_532_9)).abs() < 1e-10);
    }

    #[test]
    fn lgamma_known_values() {
        // ln Γ(1) = 0, ln Γ(2) = 0, ln Γ(3) = ln 2
        assert!(lgamma(1.0).abs() < 1e-12);
        assert!(lgamma(2.0).abs() < 1e-12);
        assert!((lgamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        // ln Γ(0.5) = ½ ln π
        let half_ln_pi = 0.5 * std::f64::consts::PI.ln();
        assert!((lgamma(0.5) - half_ln_pi).abs() < 1e-10);
    }

    #[test]
    fn trigamma_known_values() {
        let pi = std::f64::consts::PI;
        // ψ'(1) = π²/6
        let tri1 = trigamma(1.0);
        let expected1 = pi * pi / 6.0;
        assert!(
            (tri1 - expected1).abs() < 1e-10,
            "ψ'(1) = {tri1}, expected π²/6 = {expected1}"
        );
        // ψ'(2) = π²/6 − 1
        let tri2 = trigamma(2.0);
        let expected2 = pi * pi / 6.0 - 1.0;
        assert!(
            (tri2 - expected2).abs() < 1e-10,
            "ψ'(2) = {tri2}, expected π²/6 − 1 = {expected2}"
        );
    }
}
