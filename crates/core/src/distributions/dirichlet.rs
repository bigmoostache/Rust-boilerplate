//! Dirichlet distribution — closed-form computations.
//!
//! The Dirichlet is parameterized by concentration `α_k > 0` for
//! `k = 1, …, K`. It's not a standard exponential family in the
//! textbook sense (the base measure depends on the simplex constraint),
//! but we treat `α` as the "natural-like" parameters since
//! `T(x) = (ln x_1, …, ln x_K)` and the density writes
//! `p(x; α) ∝ exp(Σ (α_k − 1) ln x_k)`, giving
//! effective natural params `η_k = α_k − 1`.
//!
//! Sufficient statistics: `T(x) = (ln x_1, …, ln x_K)`, dimension `d = K`.
//! Log-partition: `A(α) = Σ ln Γ(α_k) − ln Γ(Σ α_k)`.

use nalgebra::DVector;

use super::gamma::{digamma, lgamma};

/// `E[T(x)] = (E[ln x_1], …, E[ln x_K])`.
///
/// `E[ln x_k] = ψ(α_k) − ψ(Σ α_j)`.
pub(super) fn expected_suff_stats(alpha: &[f64]) -> DVector<f64> {
    let alpha_sum: f64 = alpha.iter().sum();
    let psi_sum = digamma(alpha_sum);
    DVector::from_vec(alpha.iter().map(|&a| digamma(a) - psi_sum).collect())
}

/// Dirichlet entropy.
///
/// `H = ln B(α) + (α₀ − K) ψ(α₀) − Σ (α_k − 1) ψ(α_k)`
///
/// where `α₀ = Σ α_k` and `ln B(α) = Σ ln Γ(α_k) − ln Γ(α₀)`.
pub(super) fn entropy(alpha: &[f64]) -> f64 {
    let k = f64::from(u32::try_from(alpha.len()).unwrap_or(u32::MAX));
    let alpha_sum: f64 = alpha.iter().sum();
    let ln_beta = alpha.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_sum);
    let psi_sum = digamma(alpha_sum);
    let sum_term: f64 = alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum();
    (alpha_sum - k).mul_add(psi_sum, ln_beta) - sum_term
}

/// `E_self[ln p_other(x)]` where both are Dirichlet.
///
/// For Dirichlet, the effective natural params are `η_k = α_k − 1`,
/// so: `E_self[ln p_other] = Σ (α'_k − 1) E[ln x_k] − A(α')`
pub(super) fn cross_entropy(me_alpha: &[f64], other_alpha: &[f64]) -> f64 {
    let t = expected_suff_stats(me_alpha);
    let a_other = log_partition(other_alpha);
    let dot: f64 = other_alpha
        .iter()
        .zip(t.iter())
        .map(|(&a, ti)| (a - 1.0) * ti)
        .sum();
    dot - a_other
}

/// `A(α) = Σ ln Γ(α_k) − ln Γ(Σ α_k)`.
pub(super) fn log_partition(alpha: &[f64]) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    alpha.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: Dirichlet(α = (2, 3, 5)).
    const ALPHA: [f64; 3] = [2.0, 3.0, 5.0];

    #[test]
    fn suff_stats_dim() {
        let t = expected_suff_stats(&ALPHA);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn suff_stats_values() {
        let t = expected_suff_stats(&ALPHA);
        assert_eq!(t.len(), 3);
        let alpha_sum = 10.0;
        let psi_sum = digamma(alpha_sum);
        let t0 = t.get(0).copied().unwrap_or(f64::NAN);
        let t1 = t.get(1).copied().unwrap_or(f64::NAN);
        let t2 = t.get(2).copied().unwrap_or(f64::NAN);
        assert!((t0 - (digamma(2.0) - psi_sum)).abs() < 1e-10);
        assert!((t1 - (digamma(3.0) - psi_sum)).abs() < 1e-10);
        assert!((t2 - (digamma(5.0) - psi_sum)).abs() < 1e-10);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let ce = cross_entropy(&ALPHA, &ALPHA);
        let h = entropy(&ALPHA);
        assert!(
            (ce + h).abs() < 1e-10,
            "ce={ce}, -h={}, diff={}",
            -h,
            (ce + h).abs()
        );
    }

    #[test]
    fn log_partition_value() {
        let a = log_partition(&ALPHA);
        let expected = lgamma(2.0) + lgamma(3.0) + lgamma(5.0) - lgamma(10.0);
        assert!((a - expected).abs() < 1e-10);
    }

    #[test]
    fn uniform_on_simplex() {
        // Dirichlet(1,1,1) = Uniform on simplex
        let uniform = [1.0, 1.0, 1.0];
        // A(1,1,1) = 3·ln Γ(1) − ln Γ(3) = 0 − ln 2 = −ln 2
        let a = log_partition(&uniform);
        assert!((a + 2.0_f64.ln()).abs() < 1e-10);
    }
}
