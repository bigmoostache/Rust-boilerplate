//! Jacobi fixed-point inference with damping.
//!
//! The optimal natural parameters for each node satisfy:
//!
//! ```text
//! η_i* = (η_i^relax + Σ_k η_k^obs + Σ_j B_ij · E[T_j])
//!        / (1 + |O_i| + λ)
//! ```
//!
//! where `λ` is the entropy scale.  This is a **weighted average** in
//! natural parameter space — not an ELBO.
//!
//! Updates are fully parallel (Jacobi): all `E[T_j]` are frozen during
//! each sweep.  A double buffer avoids read–write conflicts.  Damping
//! prevents oscillations:
//!
//! ```text
//! η_i^{t+1} = (1 − α_i) · η_i^t + α_i · η_i*
//! ```

use nalgebra::DVector;

use crate::distributions::NaturalParams;
use crate::graph::Graph;
use crate::score::{Score as _, Breakdown};

/// Result of the fixed-point inference.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Maximum parameter change in the last iteration.
    pub max_change: f64,
    /// Score breakdown at convergence.
    pub score: Breakdown,
    /// Score history (one value per iteration).
    pub score_history: Vec<f64>,
}

/// Run Jacobi fixed-point inference on the graph.
///
/// Updates each node's `post` field in-place.  Returns convergence info.
///
/// # Arguments
///
/// * `graph` — the patient graph (posteriors are modified in place)
/// * `max_iter` — maximum sweeps over all nodes
/// * `tol` — convergence threshold on max parameter change
/// * `entropy_scale` — entropy scaling factor `λ` (1.0 = standard)
pub fn coordinate_ascent(
    graph: &mut Graph,
    max_iter: usize,
    tol: f64,
    entropy_scale: f64,
) -> ConvergenceResult {
    let n = graph.num_nodes();
    let mut score_history = Vec::with_capacity(max_iter);

    // ── Pre-compute per-node damping α_i ────────────────────────
    let alphas: Vec<f64> = (0..n)
        .map(|i| {
            let Some(node) = graph.nodes.get(i) else {
                return 1.0;
            };
            let num_obs = graph.observations_for(&node.name).len();
            let coupling_norm: f64 = graph
                .neighbors(&node.name)
                .iter()
                .map(|(_, mat, _)| frobenius_norm(mat))
                .sum();
            let denom = 1.0 + f64::from(u32::try_from(num_obs).unwrap_or(u32::MAX)) + coupling_norm;
            1.0 / denom
        })
        .collect();

    // ── Double buffer: E[T_i] ───────────────────────────────────
    // buffer_read is used during computation; buffer_write receives
    // the new values.  They are swapped at the end of each iteration.
    let mut buffer_read: Vec<DVector<f64>> = graph
        .nodes
        .iter()
        .map(|node| node.post.expected_suff_stats())
        .collect();
    let mut buffer_write: Vec<DVector<f64>> = buffer_read.clone();

    for iter in 0..max_iter {
        let mut max_change = 0.0_f64;

        // ── Parallel-safe sweep (Jacobi): read from buffer_read ──
        for node_idx in 0..n {
            let Some(node) = graph.nodes.get(node_idx) else {
                continue;
            };

            // Numerator: η_relax + Σ η_obs + Σ B_ij · E[T_j]
            let mut numerator = node.relax.eta_vector();

            // Add coupling contributions from neighbors (using frozen E[T_j])
            for (j_idx, coupling, transposed) in graph.neighbors(&node.name) {
                if let Some(et_j) = buffer_read.get(j_idx) {
                    let contribution = if transposed {
                        &coupling.transpose() * et_j
                    } else {
                        coupling * et_j
                    };
                    for k in 0..numerator.len().min(contribution.len()) {
                        if let (Some(dst), Some(src)) = (numerator.get_mut(k), contribution.get(k))
                        {
                            *dst += *src;
                        }
                    }
                }
            }

            // Add observation η_obs contributions
            let obs = graph.observations_for(&node.name);
            for eta_obs in obs {
                let v = eta_obs.eta_vector();
                for k in 0..numerator.len().min(v.len()) {
                    if let (Some(dst), Some(src)) = (numerator.get_mut(k), v.get(k)) {
                        *dst += *src;
                    }
                }
            }

            // Denominator: 1 + |O_i| + λ
            let num_obs = obs.len();
            let denominator =
                1.0 + f64::from(u32::try_from(num_obs).unwrap_or(u32::MAX)) + entropy_scale;

            // Fixed point: η* = numerator / denominator
            let eta_star = &numerator / denominator;

            // Damped update: η^{t+1} = (1 − α) · η^t + α · η*
            let alpha = alphas.get(node_idx).copied().unwrap_or(1.0);
            let old_eta = node.post.eta_vector();
            let new_eta = &old_eta * (1.0 - alpha) + &eta_star * alpha;

            // Track convergence
            let change = (&new_eta - &old_eta).norm();
            max_change = max_change.max(change);

            // Write new E[T_i] into buffer_write
            let reference = node.relax.clone();
            if let Some(new_post) = NaturalParams::from_eta_vector(&new_eta, &reference) {
                if let Some(et) = buffer_write.get_mut(node_idx) {
                    *et = new_post.expected_suff_stats();
                }
                if let Some(target) = graph.nodes.get_mut(node_idx) {
                    target.post = new_post;
                }
            }
        }

        // Swap buffers
        std::mem::swap(&mut buffer_read, &mut buffer_write);

        let score = graph.score_scaled(entropy_scale);
        score_history.push(score.total());

        if max_change < tol {
            return ConvergenceResult {
                converged: true,
                iterations: iter.saturating_add(1),
                max_change,
                score,
                score_history,
            };
        }
    }

    let score = graph.score_scaled(entropy_scale);
    score_history.push(score.total());

    ConvergenceResult {
        converged: false,
        iterations: max_iter,
        max_change: f64::NAN,
        score,
        score_history,
    }
}

/// Frobenius norm of a matrix: `||M||_F = sqrt(Σ m_ij²)`.
fn frobenius_norm(m: &nalgebra::DMatrix<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::distributions::NaturalParams;
    use crate::graph::{Edge, Graph, Node};

    use super::coordinate_ascent;

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

    /// Convert a Gaussian observation (value, `noise_var`) into `η_obs`.
    fn gaussian_obs(value: f64, noise_var: f64) -> NaturalParams {
        NaturalParams::Gaussian {
            eta1: value / noise_var,
            eta2: -1.0 / (2.0 * noise_var),
        }
    }

    #[test]
    fn single_node_no_obs() {
        // No observations, no edges → η* = η_relax / (1 + 0 + λ) = η_relax / 2
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);

        assert!(result.converged);
        let post = &graph.nodes.first().map(|n| &n.post);
        assert!(matches!(post, Some(NaturalParams::Gaussian { .. })));
        // η_relax = (0, -0.5). Fixed point = (0, -0.5) / 2 = (0, -0.25)
        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        let eta2 = eta
            .as_ref()
            .and_then(|v| v.get(1).copied())
            .unwrap_or(f64::NAN);
        assert!(eta1.abs() < 1e-10, "eta1={eta1}");
        assert!((eta2 + 0.25).abs() < 1e-10, "eta2={eta2}");
    }

    #[test]
    fn gaussian_observation_update() {
        // One observation: value=3, noise_var=1 → η_obs = (3, -0.5)
        // η_relax = (0, -0.5), denominator = 1 + 1 + 1 = 3
        // η* = ((0, -0.5) + (3, -0.5)) / 3 = (3, -1) / 3 = (1, -1/3)
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        graph.add_observation("A".to_owned(), gaussian_obs(3.0, 1.0));

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        let eta2 = eta
            .as_ref()
            .and_then(|v| v.get(1).copied())
            .unwrap_or(f64::NAN);
        assert!(
            (eta1 - 1.0).abs() < 1e-6,
            "eta1={eta1}, expected 1.0"
        );
        assert!(
            (eta2 - (-1.0 / 3.0)).abs() < 1e-6,
            "eta2={eta2}, expected -1/3"
        );
    }

    #[test]
    fn two_coupled_gaussians() {
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.0]),
        }];
        let mut graph = Graph::new(nodes, edges);
        graph.add_observation("A".to_owned(), gaussian_obs(5.0, 1.0));

        let result = coordinate_ascent(&mut graph, 200, 1e-8, 1.0);
        assert!(result.converged, "did not converge in 200 iterations");

        // Node A should have posterior pulled toward obs
        let eta_a = graph.nodes.first().map(|n| n.post.eta_vector());
        let a_eta1 = eta_a
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        assert!(a_eta1 > 1.0, "node A eta1={a_eta1} should be > 1");

        // Node B should be influenced by coupling (non-zero eta1)
        let eta_b = graph.nodes.get(1).map(|n| n.post.eta_vector());
        let b_eta1 = eta_b
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        assert!(
            b_eta1.abs() > 0.001,
            "node B should be influenced by coupling, eta1={b_eta1}"
        );
    }

    #[test]
    fn bernoulli_observation() {
        // Bernoulli node with flat prior η=0 (p=0.5)
        // Observe "true" → η_obs = +2 (strong positive evidence)
        let params = NaturalParams::Bernoulli { eta1: 0.0 };
        let node = Node {
            name: "test".to_owned(),
            epidemio: params.clone(),
            prev: params.clone(),
            relax: params.clone(),
            post: params,
            tau: 1.0,
        };
        let mut graph = Graph::new(vec![node], vec![]);
        graph.add_observation(
            "test".to_owned(),
            NaturalParams::Bernoulli { eta1: 2.0 },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        // η* = (0 + 2) / (1 + 1 + 1) = 2/3 ≈ 0.667
        // p = sigmoid(2/3) ≈ 0.66
        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta.as_ref().and_then(|v| v.get(0).copied()).unwrap_or(0.0);
        assert!(
            (eta1 - 2.0 / 3.0).abs() < 1e-6,
            "eta1={eta1}, expected 2/3"
        );
        let prob = 1.0 / (1.0 + (-eta1).exp());
        assert!(
            prob > 0.55,
            "p={prob}, should be > 0.55 after observing positive evidence"
        );
    }

    #[test]
    fn damping_prevents_divergence() {
        // Strong coupling that would oscillate without damping
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.0]),
        }];
        let mut graph = Graph::new(nodes, edges);
        graph.add_observation("A".to_owned(), gaussian_obs(2.0, 1.0));
        graph.add_observation("B".to_owned(), gaussian_obs(-1.0, 2.0));

        let result = coordinate_ascent(&mut graph, 300, 1e-10, 1.0);
        // Should converge thanks to damping
        assert!(
            result.converged,
            "should converge with damping, iterations={}",
            result.iterations
        );
    }
}
