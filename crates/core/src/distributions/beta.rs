//! Beta distribution — closed-form computations.
//!
//! Natural parameters: `η₁ = α − 1`, `η₂ = β − 1`.
//! Sufficient statistics: `T(x) = (ln x, ln(1−x))`.
//! Log-partition: `A(η) = ln Γ(η₁+1) + ln Γ(η₂+1) − ln Γ(η₁+η₂+2)`.

use nalgebra::DVector;

// Re-use lgamma and digamma from the gamma module.
use super::gamma::{digamma, lgamma};

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

/// `H(Beta(α,β)) = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)`.
///
/// where `ln B(α,β) = ln Γ(α) + ln Γ(β) − ln Γ(α+β)`.
pub(super) fn entropy(eta1: f64, eta2: f64) -> f64 {
    let (alpha, beta) = canonical(eta1, eta2);
    let ln_beta_fn = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta);
    let psi_sum = digamma(alpha + beta);
    (alpha + beta - 2.0).mul_add(
        psi_sum,
        ln_beta_fn - (alpha - 1.0).mul_add(digamma(alpha), (beta - 1.0) * digamma(beta)),
    )
}

/// `E_self[ln p_other(x)]` where both are Beta.
///
/// Uses direct field computation to avoid `DVector` indexing.
pub(super) fn cross_entropy(me_eta1: f64, me_eta2: f64, other_eta1: f64, other_eta2: f64) -> f64 {
    let (me_alpha, me_beta) = canonical(me_eta1, me_eta2);
    let psi_sum = digamma(me_alpha + me_beta);
    let e_ln_x = digamma(me_alpha) - psi_sum;
    let e_ln_1mx = digamma(me_beta) - psi_sum;
    let a_other = log_partition(other_eta1, other_eta2);
    other_eta1.mul_add(e_ln_x, other_eta2 * e_ln_1mx) - a_other
}

/// `A(η) = ln Γ(η₁+1) + ln Γ(η₂+1) − ln Γ(η₁+η₂+2)`.
pub(super) fn log_partition(eta1: f64, eta2: f64) -> f64 {
    let (alpha, beta) = canonical(eta1, eta2);
    lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (alpha, beta) = (2.0, 5.0);
        let psi_sum = digamma(alpha + beta);
        assert!((t[0] - (digamma(alpha) - psi_sum)).abs() < 1e-10);
        assert!((t[1] - (digamma(beta) - psi_sum)).abs() < 1e-10);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let ce = cross_entropy(ETA1, ETA2, ETA1, ETA2);
        let h = entropy(ETA1, ETA2);
        assert!((ce + h).abs() < 1e-10);
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(ETA1, ETA2);
        let expected = lgamma(2.0) + lgamma(5.0) - lgamma(7.0);
        assert!((a - expected).abs() < 1e-10);
    }
}
