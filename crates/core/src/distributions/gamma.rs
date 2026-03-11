//! Gamma distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = α − 1`, `η₂ = −β`.
//! Sufficient statistics: `T(x) = (ln x, x)`.
//! Log-partition: `A(η) = ln Γ(η₁ + 1) − (η₁ + 1) ln(−η₂)`.

use nalgebra::DVector;

/// Canonical parameters: `(α, β)`.
fn canonical(eta1: f64, eta2: f64) -> (f64, f64) {
    let alpha = eta1 + 1.0;
    let beta = -eta2;
    (alpha, beta)
}

/// Digamma function `ψ(x) = d/dx ln Γ(x)`.
///
/// Uses the asymptotic series with argument reduction.
pub(crate) fn digamma(x: f64) -> f64 {
    // Shift x up until x >= 8 for good asymptotic accuracy.
    let mut result = 0.0;
    let mut z = x;
    // Shift x up until x >= 8 for good asymptotic accuracy.
    // Shift z up until z >= 8 for good asymptotic accuracy.
    // z is always positive; at most 8 shifts are needed.
    for _ in 0..8_u32 {
        if z >= 8.0 {
            break;
        }
        result -= 1.0 / z;
        z += 1.0;
    }
    // Asymptotic expansion for large z.
    let z2 = 1.0 / (z * z);
    result += z.ln()
        - 0.5 / z
        - z2 * (1.0 / 12.0
            - z2 * (1.0 / 120.0 - z2 * (1.0 / 252.0 - z2 * (1.0 / 240.0 - z2 / 132.0))));
    result
}

/// `E[T(x)] = (E[ln x], E[x]) = (ψ(α) − ln β, α/β)`.
pub(super) fn expected_suff_stats(eta1: f64, eta2: f64) -> DVector<f64> {
    let (alpha, beta) = canonical(eta1, eta2);
    DVector::from_vec(vec![digamma(alpha) - beta.ln(), alpha / beta])
}

/// `H(Gamma(α,β)) = α − ln β + ln Γ(α) + (1 − α) ψ(α)`.
pub(super) fn entropy(eta1: f64, eta2: f64) -> f64 {
    let (alpha, beta) = canonical(eta1, eta2);
    (1.0 - alpha).mul_add(digamma(alpha), alpha - beta.ln() + lgamma(alpha))
}

/// `E_self[ln p_other(x)]` where both are Gamma.
///
/// Uses direct field computation to avoid `DVector` indexing.
pub(super) fn cross_entropy(me_eta1: f64, me_eta2: f64, other_eta1: f64, other_eta2: f64) -> f64 {
    let (me_alpha, me_beta) = canonical(me_eta1, me_eta2);
    let e_ln_x = digamma(me_alpha) - me_beta.ln();
    let e_x = me_alpha / me_beta;
    let a_other = log_partition(other_eta1, other_eta2);
    other_eta1.mul_add(e_ln_x, other_eta2 * e_x) - a_other
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
        let ce = cross_entropy(ETA1, ETA2, ETA1, ETA2);
        let h = entropy(ETA1, ETA2);
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
}
