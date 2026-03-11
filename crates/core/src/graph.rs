//! Graph structure — nodes, edges, and the patient graph.
//!
//! The graph is a Markov Random Field where each node carries a
//! distribution from an exponential family. Edges carry coupling
//! matrices `B_ij` in the sufficient-statistic space.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::distributions::NaturalParams;
use crate::observation::Observation;

/// Unique identifier for a graph node.
pub type NodeId = u32;

/// A node in the patient graph.
///
/// Each node represents a clinical variable and carries four successive
/// distribution states: epidemiological prior, previous posterior,
/// relaxed prior, and current posterior.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Human-readable name (e.g., "glycemia", "CRP", "compliance").
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
/// The coupling matrix `B_ij ∈ ℝ^{d_i × d_j}` operates in
/// sufficient-statistic space: the coupling energy is
/// `E[T_i]^T B_ij E[T_j]`.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node.
    pub i: NodeId,
    /// Target node.
    pub j: NodeId,
    /// Coupling matrix `B_ij ∈ ℝ^{d_i × d_j}`.
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
    /// Observations per node.
    pub observations: HashMap<NodeId, Vec<Observation>>,
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
        let id_to_index: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

        // Validate edges
        for edge in &edges {
            if let (Some(&i_idx), Some(&j_idx)) =
                (id_to_index.get(&edge.i), id_to_index.get(&edge.j))
            {
                if let (Some(ni), Some(nj)) = (nodes.get(i_idx), nodes.get(j_idx)) {
                    let di = ni.epidemio.suff_stat_dim();
                    let dj = nj.epidemio.suff_stat_dim();
                    debug_assert!(
                        edge.coupling.nrows() == di && edge.coupling.ncols() == dj,
                        "coupling matrix for edge ({}, {}) has shape {}×{}, expected {di}×{dj}",
                        edge.i,
                        edge.j,
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
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        let &idx = self.id_to_index.get(&id)?;
        self.nodes.get(idx)
    }

    /// Get a mutable reference to a node by its ID, or `None` if not found.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        let idx = *self.id_to_index.get(&id)?;
        self.nodes.get_mut(idx)
    }

    /// Index of a node in the `nodes` vector, or `None` if not found.
    #[must_use]
    pub fn node_index(&self, id: NodeId) -> Option<usize> {
        self.id_to_index.get(&id).copied()
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

    /// Add an observation to a node.
    ///
    /// # Panics
    ///
    /// Panics if the node ID doesn't exist.
    pub fn add_observation(&mut self, node_id: NodeId, obs: Observation) {
        debug_assert!(
            self.id_to_index.contains_key(&node_id),
            "unknown node {node_id}"
        );
        self.observations.entry(node_id).or_default().push(obs);
    }

    /// Get observations for a node (empty slice if none).
    #[must_use]
    pub fn observations_for(&self, node_id: NodeId) -> &[Observation] {
        self.observations.get(&node_id).map_or(&[], Vec::as_slice)
    }

    /// Get the list of neighbor indices and coupling matrices for a node.
    ///
    /// Returns `(neighbor_index, coupling_matrix, is_transposed)` triples.
    /// If the edge is `(i, j)` and we're asking about node `i`, returns
    /// `(j_idx, B_ij, false)`. If asking about `j`, returns
    /// `(i_idx, B_ij, true)` — meaning the coupling should be transposed.
    #[must_use]
    pub fn neighbors(&self, node_id: NodeId) -> Vec<(usize, &DMatrix<f64>, bool)> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if edge.i == node_id {
                if let Some(&j_idx) = self.id_to_index.get(&edge.j) {
                    result.push((j_idx, &edge.coupling, false));
                }
            } else if edge.j == node_id
                && let Some(&i_idx) = self.id_to_index.get(&edge.i)
            {
                result.push((i_idx, &edge.coupling, true));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian_node(id: NodeId, name: &str, mu: f64, sigma2: f64) -> Node {
        let eta1 = mu / sigma2;
        let eta2 = -1.0 / (2.0 * sigma2);
        let params = NaturalParams::Gaussian { eta1, eta2 };
        Node {
            id,
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
            make_gaussian_node(0, "A", 0.0, 1.0),
            make_gaussian_node(1, "B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            i: 0,
            j: 1,
            coupling: DMatrix::identity(2, 2),
        }];
        let graph = Graph::new(nodes, edges);
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn neighbors() {
        let nodes = vec![
            make_gaussian_node(0, "A", 0.0, 1.0),
            make_gaussian_node(1, "B", 0.0, 1.0),
            make_gaussian_node(2, "C", 0.0, 1.0),
        ];
        let edges = vec![
            Edge {
                i: 0,
                j: 1,
                coupling: DMatrix::identity(2, 2),
            },
            Edge {
                i: 1,
                j: 2,
                coupling: DMatrix::identity(2, 2),
            },
        ];
        let graph = Graph::new(nodes, edges);

        // Node 0 has 1 neighbor (node 1)
        assert_eq!(graph.neighbors(0).len(), 1);
        // Node 1 has 2 neighbors (nodes 0 and 2)
        assert_eq!(graph.neighbors(1).len(), 2);
        // Node 2 has 1 neighbor (node 1)
        assert_eq!(graph.neighbors(2).len(), 1);
    }

    #[test]
    fn observations() {
        let nodes = vec![make_gaussian_node(0, "A", 0.0, 1.0)];
        let mut graph = Graph::new(nodes, vec![]);
        assert!(graph.observations_for(0).is_empty());

        graph.add_observation(
            0,
            Observation::GaussianNoise {
                value: 1.5,
                noise_var: 0.1,
            },
        );
        assert_eq!(graph.observations_for(0).len(), 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "coupling matrix")]
    fn invalid_coupling_dims() {
        let nodes = vec![
            make_gaussian_node(0, "A", 0.0, 1.0),
            make_gaussian_node(1, "B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            i: 0,
            j: 1,
            coupling: DMatrix::identity(3, 3), // Wrong: should be 2×2
        }];
        let _graph = Graph::new(nodes, edges);
    }
}
