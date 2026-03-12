//! Coupling calibration utilities.
//!
//! Given two nodes with known canonical parameters, compute the
//! coupling coefficient `b` such that the coupling contribution
//! shifts the target node's mean by a desired amount.
//!
//! # Model
//!
//! The fixed-point update for node `a` includes:
//!
//! ```text
//! η_a* = (η_a^relax + Σ η_obs + Σ B_ab · E[T_b]) / denom
//! ```
//!
//! For a single coupling coefficient `b` at position `(i, j)` in the
//! coupling matrix, the contribution to `η_a[i]` is `b · E[T_b][j]`.
//!
//! This module computes `E[T]` for each family from canonical params,
//! then solves for `b` given a target shift on the mean of node `a`.

use crate::distributions::NaturalParams;

/// A calibration scenario for one coupling coefficient.
///
/// Describes: "When `node_a` has these params and `node_b` has those,
/// what value of `b` at position `(row, col)` would shift `node_a`'s
/// mean by `target_shift`?"
#[derive(Debug, Clone)]
pub struct CouplingScenario {
    /// Node A's canonical parameters (the node receiving the coupling).
    pub node_a: NaturalParams,
    /// Node B's canonical parameters (the node providing E[T]).
    pub node_b: NaturalParams,
    /// Row index in the coupling matrix (component of `η_a` affected).
    pub row: usize,
    /// Column index in the coupling matrix (component of `E[T_b]` used).
    pub col: usize,
    /// Number of observations on node A (affects denominator).
    pub num_obs_a: usize,
    /// Entropy scale λ (affects denominator).
    pub entropy_scale: f64,
    /// Desired shift on node A's mean.
    pub target_shift: f64,
}

/// Result of a coupling calibration.
#[derive(Debug, Clone, Copy)]
pub struct CouplingResult {
    /// The computed coupling coefficient.
    pub b: f64,
    /// `E[T_b]` at the given column index.
    pub et_b_j: f64,
    /// Current `η_a` at the given row index (from relax).
    pub eta_a_i: f64,
    /// Denominator used in the fixed-point formula.
    pub denom: f64,
    /// `η_a`[row] after applying the coupling.
    pub eta_a_i_new: f64,
    /// Node A's mean before coupling.
    pub mean_a_before: f64,
    /// Node A's mean after coupling.
    pub mean_a_after: f64,
}

/// Compute the mean of a distribution from its natural parameters.
fn compute_mean(params: &NaturalParams) -> f64 {
    match params {
        NaturalParams::Gaussian { eta1, eta2 } => {
            let sigma2 = -1.0 / (2.0 * eta2);
            eta1 * sigma2
        }
        NaturalParams::Gamma { eta1, eta2 } => {
            let alpha = eta1 + 1.0;
            let beta = -eta2;
            alpha / beta
        }
        NaturalParams::Dirichlet { eta } => {
            // For K=2 (Beta): mean = α₁/(α₁+α₂)
            // For general K: return mean of first component
            let alpha: Vec<f64> = eta.iter().map(|&e| e + 1.0).collect();
            let sum: f64 = alpha.iter().sum();
            alpha.first().copied().unwrap_or(0.0) / sum
        }
    }
}

/// Build `NaturalParams` from canonical parameters.
///
/// Helper to construct from human-readable values:
/// - Gaussian: (mu, sigma2)
/// - Gamma: (alpha, beta)
/// - Beta: (alpha, beta) → Dirichlet K=2
#[must_use]
pub fn gaussian(mu: f64, sigma2: f64) -> NaturalParams {
    NaturalParams::Gaussian {
        eta1: mu / sigma2,
        eta2: -1.0 / (2.0 * sigma2),
    }
}

/// Build Beta (Dirichlet K=2) from canonical alpha, beta.
#[must_use]
pub fn beta(alpha: f64, beta: f64) -> NaturalParams {
    NaturalParams::Dirichlet {
        eta: vec![alpha - 1.0, beta - 1.0],
    }
}

/// Build Gamma from canonical alpha, beta.
#[must_use]
pub fn gamma(alpha: f64, beta_param: f64) -> NaturalParams {
    NaturalParams::Gamma {
        eta1: alpha - 1.0,
        eta2: -beta_param,
    }
}

/// Solve for the coupling coefficient `b` in a given scenario.
///
/// The idea: node A is at its relaxed state. We want the coupling
/// contribution from node B to shift node A's mean by `target_shift`.
///
/// We compute:
/// 1. `E[T_b][col]` — the sufficient statistic of node B
/// 2. The current mean of node A (from `η_a_relax`)
/// 3. The target mean = `current_mean` + `target_shift`
/// 4. The `η_a` that would produce the target mean
/// 5. Solve for b such that the fixed-point gives that `η_a`
///
/// For now, we use a **numerical approach**: binary search on `b` to
/// find the value that produces the desired mean shift. This avoids
/// having to invert the mean-to-eta mapping analytically for each family.
#[must_use]
pub fn solve_coupling(scenario: &CouplingScenario) -> CouplingResult {
    let et_b = scenario.node_b.expected_suff_stats();
    let et_b_j = et_b.get(scenario.col).copied().unwrap_or(0.0);

    let eta_a = scenario.node_a.eta_vector();
    let eta_a_i = eta_a.get(scenario.row).copied().unwrap_or(0.0);

    let denom = 1.0
        + f64::from(u32::try_from(scenario.num_obs_a).unwrap_or(u32::MAX))
        + scenario.entropy_scale;

    let mean_before = compute_mean(&scenario.node_a);
    let target_mean = mean_before + scenario.target_shift;

    // Binary search for b
    // The coupling contribution to η_a[row] is b * et_b_j
    // η_a_new[row] = (η_a[row] + b * et_b_j) / denom
    // (ignoring other coupling terms — this is a one-edge analysis)
    //
    // But we want the FULL η to give the target mean, not just one component.
    // For simplicity, we solve in the linear regime:
    //
    // Without coupling: η_a_star = η_a_relax / denom
    // With coupling:    η_a_star[row] = (η_a_relax[row] + b * et_b_j) / denom
    //
    // We search for b that makes compute_mean(η_a_star) = target_mean.

    let dim = eta_a.len();

    // Function: given b, compute the mean of node A at the fixed point
    let mean_at_b = |b: f64| -> f64 {
        let mut eta_new = Vec::with_capacity(dim);
        for k in 0..dim {
            let base = eta_a.get(k).copied().unwrap_or(0.0);
            let coupling = if k == scenario.row { b * et_b_j } else { 0.0 };
            eta_new.push((base + coupling) / denom);
        }
        let params = match &scenario.node_a {
            NaturalParams::Gaussian { .. } => NaturalParams::Gaussian {
                eta1: *eta_new.first().unwrap_or(&0.0),
                eta2: *eta_new.get(1).unwrap_or(&-0.5),
            },
            NaturalParams::Gamma { .. } => NaturalParams::Gamma {
                eta1: *eta_new.first().unwrap_or(&0.0),
                eta2: *eta_new.get(1).unwrap_or(&-1.0),
            },
            NaturalParams::Dirichlet { .. } => NaturalParams::Dirichlet { eta: eta_new },
        };
        compute_mean(&params)
    };

    // Binary search
    let mut lo = -1e6_f64;
    let mut hi = 1e6_f64;

    // Determine search direction
    let mean_lo = mean_at_b(lo);
    let mean_hi = mean_at_b(hi);
    let increasing = mean_hi > mean_lo;

    if (increasing && target_mean > mean_hi) || (!increasing && target_mean < mean_hi) {
        hi = 1e9;
    }
    if (increasing && target_mean < mean_lo) || (!increasing && target_mean > mean_lo) {
        lo = -1e9;
    }

    for _ in 0..200_u32 {
        let mid = f64::midpoint(lo, hi);
        let mean_mid = mean_at_b(mid);
        if (mean_mid - target_mean).abs() < 1e-12 {
            break;
        }
        let go_higher = if increasing {
            mean_mid < target_mean
        } else {
            mean_mid > target_mean
        };
        if go_higher {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let b = f64::midpoint(lo, hi);
    let eta_a_i_new = b.mul_add(et_b_j, eta_a_i) / denom;
    let mean_after = mean_at_b(b);

    CouplingResult {
        b,
        et_b_j,
        eta_a_i,
        denom,
        eta_a_i_new,
        mean_a_before: mean_before,
        mean_a_after: mean_after,
    }
}

/// Write a detailed report for a coupling scenario and its result.
///
/// Uses `std::fmt::Write` so it works with `String`, `stdout`, etc.
///
/// # Errors
///
/// Returns `Err` if a write to `w` fails.
pub fn write_report<W: std::fmt::Write>(
    w: &mut W,
    name_a: &str,
    name_b: &str,
    scenario: &CouplingScenario,
    result: &CouplingResult,
) -> std::fmt::Result {
    writeln!(
        w,
        "┌─ Coupling: {name_a} ← {name_b} (B[{},{}])",
        scenario.row, scenario.col
    )?;
    writeln!(
        w,
        "│  E[T_{name_b}][{}] = {:.6}",
        scenario.col, result.et_b_j
    )?;
    writeln!(
        w,
        "│  η_{name_a}[{}] (relax) = {:.6}",
        scenario.row, result.eta_a_i
    )?;
    writeln!(w, "│  denom = {:.2}", result.denom)?;
    writeln!(w, "│")?;
    writeln!(w, "│  mean_{name_a} before = {:.4}", result.mean_a_before)?;
    writeln!(
        w,
        "│  mean_{name_a} after  = {:.4} (target shift = {:+.4})",
        result.mean_a_after, scenario.target_shift
    )?;
    writeln!(w, "│")?;
    writeln!(
        w,
        "│  ➜ B[{},{}] = {:.6}",
        scenario.row, scenario.col, result.b
    )?;
    writeln!(w, "└──────────────────────────────────────")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_self_coupling() {
        // Gaussian(mu=37, sigma2=0.25) coupled with Beta(alpha=6, beta=34)
        // When flu is at mean=0.80, how much should it shift temperature?
        let scenario = CouplingScenario {
            node_a: gaussian(37.0, 0.25),
            node_b: beta(6.0, 34.0),
            row: 0,
            col: 0,
            num_obs_a: 1,
            entropy_scale: 1.0,
            target_shift: 1.0, // shift temp mean by +1°C
        };
        let result = solve_coupling(&scenario);
        assert!(
            (result.mean_a_after - 38.0).abs() < 0.01,
            "expected mean ≈ 38, got {}",
            result.mean_a_after
        );
        assert!(result.b.is_finite(), "b should be finite");
    }

    #[test]
    fn beta_beta_coupling() {
        // Beta(alpha=6, beta=34) coupled with Beta(alpha=1.5, beta=13.5)
        // flu (mean=0.15) ← headache (mean=0.10)
        // When headache is at mean=0.90, shift flu by +0.05
        let scenario = CouplingScenario {
            node_a: beta(6.0, 34.0),
            node_b: beta(15.0, 2.0), // headache at mean ~0.88
            row: 0,
            col: 0,
            num_obs_a: 0,
            entropy_scale: 1.0,
            target_shift: 0.05,
        };
        let result = solve_coupling(&scenario);
        assert!(
            (result.mean_a_after - 0.20).abs() < 0.02,
            "expected mean ≈ 0.20, got {}",
            result.mean_a_after
        );
    }
}
