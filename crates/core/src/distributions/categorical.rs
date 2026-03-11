//! Categorical distribution — closed-form computations.
//!
//! Natural parameters: `η_k = ln(p_k / p_K)` for `k = 1, …, K−1`.
//! Sufficient statistics: `T(x) = (𝟙{x=1}, …, 𝟙{x=K−1})`.
//! Log-partition: `A(η) = ln(1 + Σ exp(η_k))`.
//!
//! The reference class `K` has implicit `η_K = 0`.

use nalgebra::DVector;

/// Recover class probabilities `(p_1, …, p_K)` from log-ratios.
///
/// Uses the log-sum-exp trick for numerical stability.
fn probabilities(eta: &[f64]) -> Vec<f64> {
    // log-sum-exp: max over all η_k and the implicit η_K = 0
    let max_eta = eta.iter().copied().fold(0.0_f64, f64::max);
    let sum_exp: f64 = eta.iter().map(|&e| (e - max_eta).exp()).sum::<f64>() + (-max_eta).exp();
    let log_z = max_eta + sum_exp.ln();
    let mut probs: Vec<f64> = eta.iter().map(|&e| (e - log_z).exp()).collect();
    probs.push((-log_z).exp()); // p_K
    probs
}

/// `E[T(x)] = (p_1, …, p_{K−1})`.
pub(super) fn expected_suff_stats(eta: &[f64]) -> DVector<f64> {
    let probs = probabilities(eta);
    // Drop the last element (p_K) — sufficient stats are for classes 1..K-1
    let stats = probs.split_last().map_or(probs.as_slice(), |(_, rest)| rest);
    DVector::from_vec(stats.to_vec())
}

/// `H(Cat) = −Σ_k p_k ln p_k`.
pub(super) fn entropy(eta: &[f64]) -> f64 {
    let probs = probabilities(eta);
    let mut h = 0.0;
    for &p in &probs {
        if p > 1e-15 {
            h -= p * p.ln();
        }
    }
    h
}

/// `E_self[ln p_other(x)]` where both are Categorical.
///
/// `= η_other · E_self[T(x)] − A(η_other)`
pub(super) fn cross_entropy(me_eta: &[f64], other_eta: &[f64]) -> f64 {
    let probs = probabilities(me_eta);
    let a_other = log_partition(other_eta);
    let dot: f64 = other_eta
        .iter()
        .zip(probs.iter())
        .map(|(o, p)| o * p)
        .sum();
    dot - a_other
}

/// `A(η) = ln(1 + Σ exp(η_k))` = `log_sum_exp(η_1, …, η_{K−1}, 0)`.
pub(super) fn log_partition(eta: &[f64]) -> f64 {
    let max_eta = eta.iter().copied().fold(0.0_f64, f64::max);
    let sum_exp: f64 = eta.iter().map(|&e| (e - max_eta).exp()).sum::<f64>() + (-max_eta).exp();
    max_eta + sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: Cat(K=3) with p = (0.2, 0.3, 0.5).
    /// η₁ = ln(0.2/0.5) = ln(0.4), η₂ = ln(0.3/0.5) = ln(0.6).
    fn standard_eta() -> Vec<f64> {
        vec![(0.2_f64 / 0.5).ln(), (0.3_f64 / 0.5).ln()]
    }

    #[test]
    fn probabilities_values() {
        let eta = standard_eta();
        let probs = probabilities(&eta);
        assert!((probs[0] - 0.2).abs() < 1e-10);
        assert!((probs[1] - 0.3).abs() < 1e-10);
        assert!((probs[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn suff_stats_values() {
        let eta = standard_eta();
        let t = expected_suff_stats(&eta);
        assert_eq!(t.len(), 2); // K-1 = 2
        assert!((t[0] - 0.2).abs() < 1e-10);
        assert!((t[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn entropy_value() {
        let eta = standard_eta();
        let h = entropy(&eta);
        let expected = -0.2 * 0.2_f64.ln() - 0.3 * 0.3_f64.ln() - 0.5 * 0.5_f64.ln();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn self_cross_entropy_equals_neg_entropy() {
        let eta = standard_eta();
        let ce = cross_entropy(&eta, &eta);
        let h = entropy(&eta);
        assert!((ce + h).abs() < 1e-10);
    }

    #[test]
    fn uniform_distribution() {
        // Uniform: all η_k = 0 → p_k = 1/K for all k
        let eta = vec![0.0, 0.0, 0.0];
        let probs = probabilities(&eta);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-10);
        }
        // H(uniform K=4) = ln 4
        let h = entropy(&eta);
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }
}
