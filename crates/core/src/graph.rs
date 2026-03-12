//! Graph structure — nodes, edges, and the patient graph.
//!
//! The graph is a Markov Random Field where each node carries a
//! distribution from an exponential family. Edges carry coupling
//! matrices `B_ij` in the sufficient-statistic space.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::distributions::NaturalParams;

/// Unique identifier for a graph node — the node's name.
pub type NodeId = String;

/// A node in the patient graph.
///
/// Each node represents a clinical variable and carries four successive
/// distribution states: epidemiological prior, previous posterior,
/// relaxed prior, and current posterior.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique name (e.g., "glycemia", "CRP", "compliance").
    pub name: String,
    /// Fixed population prior — epidemiological reference.
    pub epidemio: NaturalParams,
    /// Previous posterior — result of the last inference step.
    pub prev: NaturalParams,
    /// Relaxed prior — `prev` decayed toward `epidemio` at rate `τ`.
    pub relax: NaturalParams,
    /// Current posterior — the variable being optimized.
    pub post: NaturalParams,
    /// Relaxation time constant `τ > 0` (in the same time unit as `Δt`).
    pub tau: f64,
}

/// An edge coupling two nodes.
///
/// The coupling matrix `B_ij ∈ ℝ^{d_a × d_b}` operates in
/// sufficient-statistic space: the coupling energy is
/// `E[T_a]^T B_ab E[T_b]`.
#[derive(Debug, Clone)]
pub struct Edge {
    /// First node.
    pub node_a: NodeId,
    /// Second node.
    pub node_b: NodeId,
    /// Coupling matrix `B_ab ∈ ℝ^{d_a × d_b}`.
    pub coupling: DMatrix<f64>,
}

/// The patient graph — an MRF with observations.
#[derive(Debug, Clone)]
pub struct Graph {
    /// All nodes, indexed by their position in the vector.
    pub nodes: Vec<Node>,
    /// Node ID → index in `nodes`.
    pub id_to_index: HashMap<NodeId, usize>,
    /// All edges.
    pub edges: Vec<Edge>,
    /// Observations per node — each observation is a `NaturalParams`
    /// (`η_obs`) in the same family as the node.
    pub observations: HashMap<NodeId, Vec<NaturalParams>>,
}

impl Graph {
    /// Create a new graph from nodes and edges.
    ///
    /// Initializes `prev`, `relax`, and `post` as copies of `epidemio`
    /// for each node.
    ///
    /// # Panics
    ///
    /// Panics if edge references a non-existent node or if coupling
    /// matrix dimensions don't match sufficient-statistic dimensions.
    #[must_use]
    pub fn new(nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
        let id_to_index: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.clone(), i))
            .collect();

        // Validate edges
        for edge in &edges {
            if let (Some(&a_idx), Some(&b_idx)) =
                (id_to_index.get(&edge.node_a), id_to_index.get(&edge.node_b))
            {
                if let (Some(na), Some(nb)) = (nodes.get(a_idx), nodes.get(b_idx)) {
                    let da = na.epidemio.suff_stat_dim();
                    let db = nb.epidemio.suff_stat_dim();
                    debug_assert!(
                        edge.coupling.nrows() == da && edge.coupling.ncols() == db,
                        "coupling matrix for edge ({}, {}) has shape {}×{}, expected {da}×{db}",
                        edge.node_a,
                        edge.node_b,
                        edge.coupling.nrows(),
                        edge.coupling.ncols()
                    );
                }
            } else {
                debug_assert!(false, "edge references unknown node");
            }
        }

        Self {
            nodes,
            id_to_index,
            edges,
            observations: HashMap::new(),
        }
    }

    /// Get a node by its ID, or `None` if not found.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&Node> {
        let &idx = self.id_to_index.get(id)?;
        self.nodes.get(idx)
    }

    /// Get a mutable reference to a node by its ID, or `None` if not found.
    pub fn node_mut(&mut self, id: &str) -> Option<&mut Node> {
        let idx = *self.id_to_index.get(id)?;
        self.nodes.get_mut(idx)
    }

    /// Index of a node in the `nodes` vector, or `None` if not found.
    #[must_use]
    pub fn node_index(&self, id: &str) -> Option<usize> {
        self.id_to_index.get(id).copied()
    }

    /// Number of nodes.
    #[must_use]
    pub const fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    #[must_use]
    pub const fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Add an observation (as natural parameters `η_obs`) to a node.
    ///
    /// The observation must be a `NaturalParams` in the same family
    /// and with the same dimension as the node's distribution.
    pub fn add_observation(&mut self, node_id: NodeId, obs: NaturalParams) {
        debug_assert!(
            self.id_to_index.contains_key(&node_id),
            "unknown node {node_id}"
        );
        self.observations.entry(node_id).or_default().push(obs);
    }

    /// Get observations for a node (empty slice if none).
    #[must_use]
    pub fn observations_for(&self, node_id: &str) -> &[NaturalParams] {
        self.observations.get(node_id).map_or(&[], Vec::as_slice)
    }

    /// Get the list of neighbor indices and coupling matrices for a node.
    ///
    /// Returns `(neighbor_index, coupling_matrix, is_transposed)` triples.
    /// If the edge is `(i, j)` and we're asking about node `i`, returns
    /// `(j_idx, B_ij, false)`. If asking about `j`, returns
    /// `(i_idx, B_ij, true)` — meaning the coupling should be transposed.
    #[must_use]
    pub fn neighbors(&self, node_id: &str) -> Vec<(usize, &DMatrix<f64>, bool)> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if edge.node_a == node_id {
                if let Some(&b_idx) = self.id_to_index.get(&edge.node_b) {
                    result.push((b_idx, &edge.coupling, false));
                }
            } else if edge.node_b == node_id
                && let Some(&a_idx) = self.id_to_index.get(&edge.node_a)
            {
                result.push((a_idx, &edge.coupling, true));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn graph_creation() {
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::identity(2, 2),
        }];
        let graph = Graph::new(nodes, edges);
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn neighbors() {
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
            make_gaussian_node("C", 0.0, 1.0),
        ];
        let edges = vec![
            Edge {
                node_a: "A".to_owned(),
                node_b: "B".to_owned(),
                coupling: DMatrix::identity(2, 2),
            },
            Edge {
                node_a: "B".to_owned(),
                node_b: "C".to_owned(),
                coupling: DMatrix::identity(2, 2),
            },
        ];
        let graph = Graph::new(nodes, edges);

        // Node A has 1 neighbor (node B)
        assert_eq!(graph.neighbors("A").len(), 1);
        // Node B has 2 neighbors (nodes A and C)
        assert_eq!(graph.neighbors("B").len(), 2);
        // Node C has 1 neighbor (node B)
        assert_eq!(graph.neighbors("C").len(), 1);
    }

    #[test]
    fn observations() {
        let nodes = vec![make_gaussian_node("A", 0.0, 1.0)];
        let mut graph = Graph::new(nodes, vec![]);
        assert!(graph.observations_for("A").is_empty());

        graph.add_observation(
            "A".to_owned(),
            NaturalParams::Gaussian {
                eta1: 1.5 / 0.1,
                eta2: -1.0 / (2.0 * 0.1),
            },
        );
        assert_eq!(graph.observations_for("A").len(), 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "coupling matrix")]
    fn invalid_coupling_dims() {
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::identity(3, 3), // Wrong: should be 2×2
        }];
        let _graph = Graph::new(nodes, edges);
    }
}
