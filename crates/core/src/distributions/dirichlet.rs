//! Dirichlet distribution — closed-form computations.
//!
//! The Dirichlet is parameterized by concentration `α_k > 0` for
//! `k = 1, …, K`. We store the **true natural parameters**
//! `η_k = α_k − 1`, so `α_k = η_k + 1`.
//!
//! Sufficient statistics: `T(x) = (ln x_1, …, ln x_K)`, dimension `d = K`.
//! Log-partition: `A(η) = Σ ln Γ(η_k + 1) − ln Γ(Σ (η_k + 1))`.
//! Base measure: `h(x) = 1_{x ∈ simplex}` — since it doesn't depend
//! on `η`, `E[ln h(x)] = 0`.

use nalgebra::DVector;

use super::gamma::{digamma, lgamma};

use super::exp_family::ExponentialFamily;

/// Marker type for the Dirichlet exponential family.
pub(crate) struct DirichletDist;

impl ExponentialFamily for DirichletDist {
    fn log_partition(eta: &DVector<f64>) -> f64 {
        let alpha: Vec<f64> = eta.iter().map(|&e| e + 1.0).collect();
        log_partition(&alpha)
    }

    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64> {
        let alpha: Vec<f64> = eta.iter().map(|&e| e + 1.0).collect();
        expected_suff_stats(&alpha)
    }

    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// `E[T(x)] = (E[ln x_1], …, E[ln x_K])`.
///
/// `E[ln x_k] = ψ(α_k) − ψ(Σ α_j)`.
pub(super) fn expected_suff_stats(alpha: &[f64]) -> DVector<f64> {
    let alpha_sum: f64 = alpha.iter().sum();
    let psi_sum = digamma(alpha_sum);
    DVector::from_vec(alpha.iter().map(|&a| digamma(a) - psi_sum).collect())
}

/// `A(α) = Σ ln Γ(α_k) − ln Γ(Σ α_k)`.
pub(super) fn log_partition(alpha: &[f64]) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    alpha.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::exp_family::{cross_entropy as ef_ce, entropy as ef_h};

    /// Reference: Dirichlet(α = (2, 3, 5)) → η = (1, 2, 4).
    const ALPHA: [f64; 3] = [2.0, 3.0, 5.0];
    const ETA: [f64; 3] = [1.0, 2.0, 4.0]; // α − 1

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
        let eta = DVector::from_vec(ETA.to_vec());
        let ce = ef_ce::<DirichletDist>(&eta, &eta);
        let h = ef_h::<DirichletDist>(&eta);
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
