//! YAML types for calibration statements.
//!
//! A calibration file declares, for each edge `(node_a, node_b)`, a
//! list of statements that constrain the coupling matrix `B`.  The
//! pipeline resolves node families from the graph definition, computes
//! `T_B(x_B)`, assembles a linear system, and solves for `B`.

use serde::Deserialize;

use crate::schema::raw::{FamilyDef, GraphConfig};

/// Top-level calibration document.
///
/// ```yaml
/// calibration:
///   - edge: [temperature, flu]
///     statements:
///       - type: conditional_mean
///         x_b: [0.99, 0.01]
///         target_mean: 39.5
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationConfig {
    /// One entry per edge to calibrate.
    pub calibration: Vec<EdgeCalibrationDef>,
}

/// Calibration block for a single edge.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCalibrationDef {
    /// `[node_a, node_b]` — the coupling direction is `node_a ← node_b`.
    pub edge: (String, String),
    /// Calibration statements constraining the coupling matrix.
    pub statements: Vec<StatementDef>,
}

/// A single calibration constraint.
///
/// Each statement fixes one or more entries of the coupling matrix `B`
/// by expressing a condition on the conditional distribution
/// `p(x_A | x_B = v)`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum StatementDef {
    /// `E[X_A | X_B = x_b] = target_mean`.
    ///
    /// Requires a matching `variance_unchanged` or
    /// `concentration_unchanged` statement at the same `x_b` to make
    /// the system linear.
    ///
    /// # Constraint count
    ///
    /// - Gaussian / Gamma `A`: 1 equation.
    /// - Dirichlet `K` `A`: `K − 1` equations.
    #[serde(rename = "conditional_mean")]
    ConditionalMean {
        /// Point in the sample space of node B.
        ///
        /// - Gaussian B: a single float `[v]` or just `v`.
        /// - Gamma B: a single positive float `[v]`.
        /// - Dirichlet K B: a K-vector on the simplex `[p₁, …, p_K]`.
        ///
        /// Entries of exactly `0.0` produce `T_B_j = −∞` (for log-based
        /// families), which forces the corresponding column of `B` to zero.
        x_b: ObsPoint,
        /// Target conditional mean.
        ///
        /// - Gaussian / Gamma `A`: a single float.
        /// - Dirichlet `K` `A`: a K-vector of probabilities
        ///   (only `K − 1` are independent; the last is implied).
        target_mean: TargetMean,
    },

    /// `V[X_A | X_B = x_b] = V[X_A]` (variance unchanged).
    ///
    /// For Gaussian `A`: forces `B[1,:] · T_B(x_b) = 0` (1 equation).
    /// For Gamma `A`: forces `B[1,:] · T_B(x_b) = 0` (1 equation).
    ///
    /// For Dirichlet `A`, use `concentration_unchanged` instead.
    #[serde(rename = "variance_unchanged")]
    VarianceUnchanged {
        /// Point in the sample space of node B.
        x_b: ObsPoint,
    },

    /// `Σ α_k^cond = Σ α_k` (total concentration unchanged).
    ///
    /// Only valid when node A is Dirichlet.
    /// Forces `Σ_k B[k,:] · T_B(x_b) = 0` — one equation per `d_B`
    /// column (actually one scalar equation: the sum of all rows'
    /// dot products with `T_B` equals zero).
    #[serde(rename = "concentration_unchanged")]
    ConcentrationUnchanged {
        /// Point in the sample space of node B.
        x_b: ObsPoint,
    },

    /// `B · T_B(x_b) = 0` — no effect at the given point.
    ///
    /// Gives `d_A` equations (one per row of `B`).
    ///
    /// If `T_B(x_b)` has `±∞` entries (e.g. `log(0)`), the
    /// corresponding columns of `B` are forced to zero instead of
    /// producing linear equations.
    #[serde(rename = "no_effect")]
    NoEffect {
        /// Point in the sample space of node B.
        x_b: ObsPoint,
    },

    /// Force a single entry `B[row, col] = 0`.
    #[serde(rename = "entry_zero")]
    EntryZero {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
    },
}

/// A point in the sample space of a node.
///
/// Untagged so YAML can use bare values:
/// - `x_b: 39.5` (Gaussian/Gamma scalar)
/// - `x_b: [0.99, 0.01]` (Dirichlet simplex)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ObsPoint {
    /// Scalar value (for Gaussian / Gamma nodes).
    Scalar(f64),
    /// Vector on the simplex (for Dirichlet nodes).
    Vec(Vec<f64>),
}

/// Target conditional mean.
///
/// Untagged for convenient YAML:
/// - `target_mean: 39.5` (Gaussian/Gamma scalar)
/// - `target_mean: [0.6, 0.4]` (Dirichlet proportions)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TargetMean {
    /// Scalar mean (Gaussian / Gamma).
    Scalar(f64),
    /// Mean vector (Dirichlet K).
    Vec(Vec<f64>),
}

// ---------------------------------------------------------------------------
// Resolved types (after matching with graph)
// ---------------------------------------------------------------------------

/// A resolved edge calibration: families are known, dimensions computed.
#[derive(Debug)]
pub struct ResolvedEdge {
    /// Node A name.
    pub name_a: String,
    /// Node B name.
    pub name_b: String,
    /// Family of node A (canonical params = prior).
    pub family_a: FamilyDef,
    /// Family of node B (canonical params = prior).
    pub family_b: FamilyDef,
    /// Dimension of `T_A` (= number of rows in `B`).
    pub dim_a: usize,
    /// Dimension of `T_B` (= number of columns in `B`).
    pub dim_b: usize,
    /// The calibration statements.
    pub statements: Vec<StatementDef>,
}

/// Resolve calibration edges against a graph config.
///
/// Looks up node families from the graph and computes dimensions.
///
/// # Errors
///
/// Returns an error string if:
/// - A referenced node doesn't exist in the graph.
/// - The graph has no nodes.
pub fn resolve_edges(
    cal: &CalibrationConfig,
    graph: &GraphConfig,
) -> Result<Vec<ResolvedEdge>, String> {
    let mut resolved = Vec::with_capacity(cal.calibration.len());

    for edge_cal in &cal.calibration {
        let (ref name_a, ref name_b) = edge_cal.edge;

        let family_a = find_family(name_a, graph)
            .ok_or_else(|| format!("node '{name_a}' not found in graph"))?;
        let family_b = find_family(name_b, graph)
            .ok_or_else(|| format!("node '{name_b}' not found in graph"))?;

        let dim_a = family_dim(&family_a);
        let dim_b = family_dim(&family_b);

        resolved.push(ResolvedEdge {
            name_a: name_a.clone(),
            name_b: name_b.clone(),
            family_a,
            family_b,
            dim_a,
            dim_b,
            statements: edge_cal.statements.clone(),
        });
    }

    Ok(resolved)
}

/// Find a node's family definition in the graph config.
fn find_family(name: &str, graph: &GraphConfig) -> Option<FamilyDef> {
    graph
        .nodes
        .iter()
        .find(|n| n.name == name)
        .map(|n| n.family.clone())
}

/// Dimension of the sufficient statistic vector for a family.
#[must_use]
pub const fn family_dim(family: &FamilyDef) -> usize {
    match family {
        FamilyDef::Gaussian { .. } | FamilyDef::Gamma { .. } | FamilyDef::Beta { .. } => 2,
        FamilyDef::Dirichlet { alpha } => alpha.len(),
    }
}
