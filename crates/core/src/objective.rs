//! ELBO (Evidence Lower Bound) computation.
//!
//! The variational objective has four terms:
//!
//! ```text
//! F = Σ_{(i,j)} E[T_i]^T B_ij E[T_j]     (couplings)
//!   + Σ_i E_post[ln p_relax(x_i)]          (relaxed priors)
//!   + Σ_i Σ_k E_post[ln p(obs_k | x_i)]   (observations)
//!   + Σ_i H(post_i)                         (entropy)
//! ```
//!
//! All terms are **fully analytical** for exponential families.

use crate::graph::Graph;

/// Detailed breakdown of the ELBO into its four components.
#[derive(Debug, Clone, Copy)]
pub struct ElboBreakdown {
    /// Coupling energy: `Σ E[T_i]^T B_ij E[T_j]`.
    pub coupling: f64,
    /// Prior term: `Σ E_post[ln p_relax(x_i)]`.
    pub prior: f64,
    /// Observation term: `Σ Σ E_post[ln p(obs | x)]`.
    pub observation: f64,
    /// Entropy term: `Σ H(post_i)`.
    pub entropy: f64,
}

impl ElboBreakdown {
    /// Total ELBO value `F = coupling + prior + observation + entropy`.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.coupling + self.prior + self.observation + self.entropy
    }
}

/// ELBO computation for the patient graph.
pub trait Elbo {
    /// Compute the full ELBO and its breakdown.
    fn elbo(&self) -> ElboBreakdown;
}

impl Elbo for Graph {
    fn elbo(&self) -> ElboBreakdown {
        ElboBreakdown {
            coupling: coupling_term(self),
            prior: prior_term(self),
            observation: observation_term(self),
            entropy: entropy_term(self),
        }
    }
}

/// Coupling energy: `Σ_{(i,j)} E[T_i]^T B_ij E[T_j]`.
fn coupling_term(graph: &Graph) -> f64 {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            let i_idx = graph.node_index(&edge.i)?;
            let j_idx = graph.node_index(&edge.j)?;
            let ti = graph.nodes.get(i_idx)?.post.expected_suff_stats();
            let tj = graph.nodes.get(j_idx)?.post.expected_suff_stats();
            let product = ti.transpose() * &edge.coupling * tj;
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

/// Observation term: `Σ_i Σ_k E_post[ln p(obs_k | x_i)]`.
fn observation_term(graph: &Graph) -> f64 {
    graph
        .nodes
        .iter()
        .map(|node| {
            graph
                .observations_for(&node.name)
                .iter()
                .map(|obs| obs.expected_log_likelihood(&node.post))
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
    use crate::observation::Observation;

    use super::Elbo as _;

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
    fn elbo_no_edges_no_obs() {
        // Single node, prior == posterior → prior term = −H, so F = 0
        let graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        let elbo = graph.elbo();

        assert!(elbo.coupling.abs() < 1e-12);
        assert!(elbo.observation.abs() < 1e-12);
        // prior = E_post[ln p_relax] = −H when post == relax
        // entropy = H
        // so prior + entropy = 0
        assert!(
            (elbo.prior + elbo.entropy).abs() < 1e-10,
            "prior={}, entropy={}, sum={}",
            elbo.prior,
            elbo.entropy,
            elbo.prior + elbo.entropy
        );
    }

    #[test]
    fn elbo_with_coupling() {
        // Two Gaussian nodes with identity coupling
        let nodes = vec![
            make_gaussian_node("A", 1.0, 1.0),
            make_gaussian_node("B", 2.0, 1.0),
        ];
        let edges = vec![Edge {
            i: "A".to_owned(),
            j: "B".to_owned(),
            coupling: DMatrix::identity(2, 2),
        }];
        let graph = Graph::new(nodes, edges);
        let elbo = graph.elbo();

        // Coupling = E[T_A]^T · I · E[T_B]
        // = E[x_A]·E[x_B] + E[x_A²]·E[x_B²]
        // = 1·2 + (1+1)·(4+1) = 2 + 10 = 12
        assert!(
            (elbo.coupling - 12.0).abs() < 1e-10,
            "coupling={}",
            elbo.coupling
        );
    }

    #[test]
    fn elbo_with_observation() {
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 0.0,
                noise_var: 1.0,
            },
        );
        let elbo = graph.elbo();

        // Obs = E[ln N(0; x, 1)] = −½ ln(2π) − E[(0−x)²]/2
        // = −½ ln(2π) − (0 + 0 + σ²)/2 = −½ ln(2π) − ½
        let expected_obs = (-0.5f64).mul_add((2.0 * std::f64::consts::PI).ln(), -0.5);
        assert!(
            (elbo.observation - expected_obs).abs() < 1e-10,
            "obs={}, expected={}",
            elbo.observation,
            expected_obs
        );
    }

    #[test]
    fn elbo_breakdown_total() {
        let nodes = vec![
            make_gaussian_node("A", 1.0, 1.0),
            make_gaussian_node("B", 2.0, 1.0),
        ];
        let edges = vec![Edge {
            i: "A".to_owned(),
            j: "B".to_owned(),
            coupling: DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]),
        }];
        let mut graph = Graph::new(nodes, edges);
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 0.5,
                noise_var: 0.5,
            },
        );
        let elbo = graph.elbo();
        let manual_total = elbo.coupling + elbo.prior + elbo.observation + elbo.entropy;
        assert!(
            (elbo.total() - manual_total).abs() < 1e-15,
            "total={}, manual={}",
            elbo.total(),
            manual_total
        );
    }
}
