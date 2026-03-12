//! Variational score computation.
//!
//! The scoring functional has four terms:
//!
//! ```text
//! F = Σ_{(i,j)} E[T_i]^T B_ij E[T_j]     (couplings)
//!   + Σ_i E_post[ln p_relax(x_i)]          (relaxed priors)
//!   + Σ_i Σ_k E_post[ln p_obs_k(x_i)]     (observations)
//!   + λ · Σ_i H(post_i)                    (entropy, scaled)
//! ```
//!
//! Observations are `NaturalParams` (`η_obs`) in the same family as
//! their node, so the observation term reduces to a cross-entropy.
//!
//! All terms are **fully analytical** for exponential families.

use crate::graph::Graph;

/// Detailed breakdown of the variational score into its four components.
#[derive(Debug, Clone, Copy)]
pub struct Breakdown {
    /// Coupling energy: `Σ E[T_i]^T B_ij E[T_j]`.
    pub coupling: f64,
    /// Prior term: `Σ E_post[ln p_relax(x_i)]`.
    pub prior: f64,
    /// Observation term: `Σ Σ E_post[ln p_obs(x)]`.
    pub observation: f64,
    /// Entropy term: `λ · Σ H(post_i)`.
    pub entropy: f64,
}

impl Breakdown {
    /// Total score `F = coupling + prior + observation + entropy`.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.coupling + self.prior + self.observation + self.entropy
    }
}

/// Variational score computation for the patient graph.
pub trait Score {
    /// Compute the full score and its breakdown with standard entropy (λ=1).
    fn score(&self) -> Breakdown;

    /// Compute the score with a scaled entropy term.
    ///
    /// `entropy_scale = 1.0` gives standard variational inference.
    /// Higher values produce softer (more uncertain) posteriors.
    fn score_scaled(&self, entropy_scale: f64) -> Breakdown;
}

impl Score for Graph {
    fn score(&self) -> Breakdown {
        self.score_scaled(1.0)
    }

    fn score_scaled(&self, entropy_scale: f64) -> Breakdown {
        Breakdown {
            coupling: coupling_term(self),
            prior: prior_term(self),
            observation: observation_term(self),
            entropy: entropy_scale * entropy_term(self),
        }
    }
}

/// Coupling energy: `Σ_{(i,j)} E[T_i]^T B_ij E[T_j]`.
fn coupling_term(graph: &Graph) -> f64 {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            let a_idx = graph.node_index(&edge.node_a)?;
            let b_idx = graph.node_index(&edge.node_b)?;
            let ta = graph.nodes.get(a_idx)?.post.expected_suff_stats();
            let tb = graph.nodes.get(b_idx)?.post.expected_suff_stats();
            let product = ta.transpose() * &edge.coupling * tb;
            product.get((0, 0)).copied()
        })
        .sum()
}

/// Prior term: `Σ_i E_post[ln p_relax(x_i)]`.
fn prior_term(graph: &Graph) -> f64 {
    graph
        .nodes
        .iter()
        .map(|node| node.post.cross_entropy(&node.relax))
        .sum()
}

/// Observation term: `Σ_i Σ_k E_post[ln p_obs_k(x_i)]`.
///
/// Each observation `η_obs` is a `NaturalParams` in the same family as
/// the node, so `E_post[ln p_obs]` is the cross-entropy.
fn observation_term(graph: &Graph) -> f64 {
    graph
        .nodes
        .iter()
        .map(|node| {
            graph
                .observations_for(&node.name)
                .iter()
                .map(|eta_obs| node.post.cross_entropy(eta_obs))
                .sum::<f64>()
        })
        .sum()
}

/// Entropy term: `Σ_i H(post_i)`.
fn entropy_term(graph: &Graph) -> f64 {
    graph.nodes.iter().map(|node| node.post.entropy()).sum()
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::distributions::NaturalParams;
    use crate::graph::{Edge, Graph, Node};

    use super::Score as _;

    fn make_gaussian_node(name: &str, mu: f64, sigma2: f64) -> Node {
        let eta1 = mu / sigma2;
        let eta2 = -1.0 / (2.0 * sigma2);
        let params = NaturalParams::Gaussian { eta1, eta2 };
        Node {
            name: name.to_owned(),
            epidemio: params.clone(),
            prev: params.clone(),
            relax: params.clone(),
            post: params,
            tau: 1.0,
        }
    }

    #[test]
    fn score_no_edges_no_obs() {
        // Single node, prior == posterior → prior term = −H, so F = 0
        let graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        let score = graph.score();

        assert!(score.coupling.abs() < 1e-12);
        assert!(score.observation.abs() < 1e-12);
        // prior = E_post[ln p_relax] = −H when post == relax
        // entropy = H
        // so prior + entropy = 0
        assert!(
            (score.prior + score.entropy).abs() < 1e-10,
            "prior={}, entropy={}, sum={}",
            score.prior,
            score.entropy,
            score.prior + score.entropy
        );
    }

    #[test]
    fn score_with_coupling() {
        // Two Gaussian nodes with identity coupling
        let nodes = vec![
            make_gaussian_node("A", 1.0, 1.0),
            make_gaussian_node("B", 2.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::identity(2, 2),
        }];
        let graph = Graph::new(nodes, edges);
        let score = graph.score();

        // Coupling = E[T_A]^T · I · E[T_B]
        // = E[x_A]·E[x_B] + E[x_A²]·E[x_B²]
        // = 1·2 + (1+1)·(4+1) = 2 + 10 = 12
        assert!(
            (score.coupling - 12.0).abs() < 1e-10,
            "coupling={}",
            score.coupling
        );
    }

    #[test]
    fn score_with_observation() {
        // Observation is η_obs in same family → cross-entropy
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        // η_obs for observing value=0 with noise_var=1: (0/1, -1/2) = (0, -0.5)
        graph.add_observation(
            "A".to_owned(),
            NaturalParams::Gaussian {
                eta1: 0.0,
                eta2: -0.5,
            },
        );
        let score = graph.score();

        // Obs term = E_post[ln p_obs(x)] = cross_entropy(post, obs)
        // When post == obs: cross_entropy = -H
        // So obs = -H, and prior + entropy = 0, total = coupling + obs = -H
        assert!(
            score.observation < 0.0,
            "obs={}, should be negative",
            score.observation
        );
    }

    #[test]
    fn score_breakdown_total() {
        let nodes = vec![
            make_gaussian_node("A", 1.0, 1.0),
            make_gaussian_node("B", 2.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]),
        }];
        let mut graph = Graph::new(nodes, edges);
        graph.add_observation(
            "A".to_owned(),
            NaturalParams::Gaussian {
                eta1: 0.5 / 0.5,
                eta2: -1.0 / (2.0 * 0.5),
            },
        );
        let score = graph.score();
        let manual_total = score.coupling + score.prior + score.observation + score.entropy;
        assert!(
            (score.total() - manual_total).abs() < 1e-15,
            "total={}, manual={}",
            score.total(),
            manual_total
        );
    }
}
