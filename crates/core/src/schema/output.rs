//! YAML output types for inference results.
//!
//! These types are serialized to produce human-readable YAML output
//! after running inference.  Each node's posterior is reported in both
//! canonical (human-readable) and natural parameter form.

use serde::Serialize;

use crate::graph::Graph;
use crate::inference::ConvergenceResult;

use super::raw::FamilyDef;

// ---------------------------------------------------------------------------
// Top-level output
// ---------------------------------------------------------------------------

/// Complete inference output, serializable to YAML.
#[derive(Debug, Serialize)]
pub struct InferenceResult {
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Number of coordinate-ascent iterations performed.
    pub iterations: usize,
    /// Maximum parameter change in the final iteration.
    pub max_change: f64,
    /// Score breakdown at convergence.
    pub score: ScoreBreakdownYaml,
    /// Score value at the end of each iteration.
    pub score_history: Vec<f64>,
    /// Posterior state for each node.
    pub posteriors: Vec<NodePosterior>,
}

/// Score breakdown in the output.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ScoreBreakdownYaml {
    /// Total score = coupling + prior + observation + entropy.
    pub total: f64,
    /// Coupling energy term.
    pub coupling: f64,
    /// Prior (relaxed) term.
    pub prior: f64,
    /// Observation log-likelihood term.
    pub observation: f64,
    /// Entropy term.
    pub entropy: f64,
}

/// Posterior state of a single node.
#[derive(Debug, Serialize)]
pub struct NodePosterior {
    /// Human-readable node name.
    pub name: String,
    /// Posterior in canonical (human-readable) parameters.
    pub family: FamilyDef,
    /// Raw natural parameters as a flat vector.
    pub natural_params: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Build the result structure from inference results and the graph.
#[must_use]
pub fn build_result(graph: &Graph, result: &ConvergenceResult) -> InferenceResult {
    let posteriors: Vec<NodePosterior> = graph
        .nodes
        .iter()
        .map(|node| NodePosterior {
            name: node.name.clone(),
            family: node.post.to_canonical(),
            natural_params: node.post.eta_vector().as_slice().to_vec(),
        })
        .collect();

    InferenceResult {
        converged: result.converged,
        iterations: result.iterations,
        max_change: result.max_change,
        score: ScoreBreakdownYaml {
            total: result.score.total(),
            coupling: result.score.coupling,
            prior: result.score.prior,
            observation: result.score.observation,
            entropy: result.score.entropy,
        },
        score_history: result.score_history.clone(),
        posteriors,
    }
}

/// Serialize inference output to a YAML string.
///
/// # Errors
///
/// Returns a YAML serialization error if the output cannot be serialized
/// (should never happen for well-formed data).
pub fn to_yaml(output: &InferenceResult) -> Result<String, serde_yaml::Error> {
    serde_yaml::to_string(output)
}
