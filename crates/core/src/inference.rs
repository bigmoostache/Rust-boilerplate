//! Mean-field coordinate ascent inference.
//!
//! For exponential families under mean-field factorization, the optimal
//! update for each node's posterior natural parameters is:
//!
//! ```text
//! η_i^{new} = η_i^{relax}
//!           + Σ_{j ∈ N(i)} B_ij · E[T_j]   (coupling from neighbors)
//!           + Σ_k ∇_η E[ln p(obs_k | x)]    (observation gradient)
//! ```
//!
//! This gives a **closed-form** per-node update — no inner optimization loop.
//! We sweep over all nodes repeatedly until convergence.

use nalgebra::DVector;

use crate::distributions::NaturalParams;
use crate::graph::Graph;
use crate::objective::{Elbo as _, ElboBreakdown};
use crate::observation::Observation;

/// Result of the coordinate ascent inference.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Maximum parameter change in the last iteration.
    pub max_change: f64,
    /// ELBO breakdown at convergence.
    pub elbo: ElboBreakdown,
    /// ELBO history (one value per iteration).
    pub elbo_history: Vec<f64>,
}

/// Run mean-field coordinate ascent on the graph.
///
/// Updates each node's `post` field in-place.  Returns convergence info.
///
/// # Arguments
///
/// * `graph` — the patient graph (posteriors are modified in place)
/// * `max_iter` — maximum sweeps over all nodes
/// * `tol` — convergence threshold on max parameter change
/// * `entropy_scale` — entropy scaling factor `λ` (1.0 = standard VI)
pub fn coordinate_ascent(
    graph: &mut Graph,
    max_iter: usize,
    tol: f64,
    entropy_scale: f64,
) -> ConvergenceResult {
    let mut elbo_history = Vec::with_capacity(max_iter);

    for iter in 0..max_iter {
        let mut max_change = 0.0_f64;

        for node_idx in 0..graph.num_nodes() {
            let new_eta = compute_optimal_eta(graph, node_idx, entropy_scale);
            if let Some(node) = graph.nodes.get(node_idx) {
                let old_eta = node.post.eta_vector();
                let change = (&new_eta - &old_eta).norm();
                max_change = max_change.max(change);
                let reference = node.relax.clone();
                if let Some(new_post) = NaturalParams::from_eta_vector(&new_eta, &reference)
                    && let Some(target) = graph.nodes.get_mut(node_idx)
                {
                    target.post = new_post;
                }
            }
        }

        let elbo = graph.elbo_scaled(entropy_scale);
        elbo_history.push(elbo.total());

        if max_change < tol {
            return ConvergenceResult {
                converged: true,
                iterations: iter.saturating_add(1),
                max_change,
                elbo,
                elbo_history,
            };
        }
    }

    let elbo = graph.elbo_scaled(entropy_scale);
    elbo_history.push(elbo.total());

    ConvergenceResult {
        converged: false,
        iterations: max_iter,
        max_change: f64::NAN,
        elbo,
        elbo_history,
    }
}

/// Compute the optimal natural parameters for node `node_idx`,
/// given all other nodes' current posteriors.
///
/// With entropy scale `λ`, the update becomes:
///
/// ```text
/// η_i^{new} = (1/λ) · (η_i^{relax} + Σ_{j} coupling_j + obs_grad)
/// ```
///
/// When `λ = 1`, this is standard mean-field variational inference.
fn compute_optimal_eta(graph: &Graph, node_idx: usize, entropy_scale: f64) -> DVector<f64> {
    let Some(node) = graph.nodes.get(node_idx) else {
        return DVector::zeros(0);
    };

    // Start with the relaxed prior's natural parameters
    let mut eta = node.relax.eta_vector();

    // Add coupling contributions from all neighbors
    let neighbors = graph.neighbors(&node.name);
    for (j_idx, coupling, transposed) in neighbors {
        if let Some(neighbor) = graph.nodes.get(j_idx) {
            let tj = neighbor.post.expected_suff_stats();
            let contribution = if transposed {
                &coupling.transpose() * &tj
            } else {
                coupling * &tj
            };
            // Safe element-wise addition (dimensions guaranteed by graph construction)
            for i in 0..eta.len().min(contribution.len()) {
                if let (Some(dst), Some(src)) = (eta.get_mut(i), contribution.get(i)) {
                    *dst += *src;
                }
            }
        }
    }

    // Add observation contributions
    let obs = graph.observations_for(&node.name);
    if !obs.is_empty() {
        let obs_gradient = observation_gradient(&node.post, obs);
        for i in 0..eta.len().min(obs_gradient.len()) {
            if let (Some(dst), Some(src)) = (eta.get_mut(i), obs_gradient.get(i)) {
                *dst += *src;
            }
        }
    }

    // Apply entropy scaling: η_new = (1/λ) · (η_relax + coupling + obs)
    // When λ = 0 (no entropy term), skip scaling — the optimal update
    // is the raw sum of prior + coupling + observation contributions.
    if entropy_scale.abs() > f64::EPSILON && (entropy_scale - 1.0).abs() > f64::EPSILON {
        let inv_lambda = 1.0 / entropy_scale;
        for i in 0..eta.len() {
            if let Some(v) = eta.get_mut(i) {
                *v *= inv_lambda;
            }
        }
    }

    eta
}

/// Compute the gradient of observation log-likelihood w.r.t. natural
/// parameters: `Σ_k ∇_η E_η[ln p(obs_k | x)]`.
///
/// For conjugate observation models, this has a closed form.
/// For non-conjugate models, we approximate via the delta method.
fn observation_gradient(post: &NaturalParams, observations: &[Observation]) -> DVector<f64> {
    let dim = post.suff_stat_dim();
    let mut grad = DVector::zeros(dim);

    for obs in observations {
        let obs_grad = single_observation_gradient(post, obs);
        grad += obs_grad;
    }

    grad
}

/// Gradient of a single observation's expected log-likelihood w.r.t.
/// the node's natural parameters.
///
/// For Gaussian node + Gaussian noise observation, this is exact:
/// ```text
/// ∇_η E_η[ln N(obs; x, σ²_n)] = (obs/σ²_n, −1/(2σ²_n))
/// ```
///
/// For other family/observation combos, we use the identity:
/// ```text
/// ∇_η E_η[f(x)] = Cov_η[T(x), f(x)]
/// ```
/// approximated via the delta method when needed.
fn single_observation_gradient(post: &NaturalParams, obs: &Observation) -> DVector<f64> {
    match (post, obs) {
        // Gaussian node + Gaussian noise: exact conjugate update
        (NaturalParams::Gaussian { .. }, Observation::GaussianNoise { value, noise_var }) => {
            // ln p(obs|x) = −½ ln(2πσ²_n) − (obs−x)²/(2σ²_n)
            // = const + obs·x/σ²_n − x²/(2σ²_n) + const
            // So the natural-parameter contribution is (obs/σ²_n, −1/(2σ²_n))
            DVector::from_vec(vec![value / noise_var, -1.0 / (2.0 * noise_var)])
        }

        // Poisson node + Poisson count: exact conjugate
        (NaturalParams::Poisson { eta1 }, Observation::PoissonCount { count }) => {
            let k = f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
            DVector::from_vec(vec![k - eta1.exp()])
        }

        // All other combos (noisy channel, noisy categorical, etc.):
        // numerical gradient via finite differences
        _ => numerical_observation_gradient(post, obs),
    }
}

/// Numerical gradient of `E_η[ln p(obs|x)]` w.r.t. natural parameters.
///
/// Uses central finite differences with step `h = 1e-6`.
/// This is the fallback for non-conjugate (family, observation) pairs.
fn numerical_observation_gradient(post: &NaturalParams, obs: &Observation) -> DVector<f64> {
    let eta = post.eta_vector();
    let dim = eta.len();
    let h = 1e-6;
    let mut grad = DVector::zeros(dim);

    for k in 0..dim {
        let mut eta_plus = eta.clone();
        let mut eta_minus = eta.clone();
        if let Some(ep) = eta_plus.get_mut(k) {
            *ep += h;
        }
        if let Some(em) = eta_minus.get_mut(k) {
            *em -= h;
        }

        let post_plus = NaturalParams::from_eta_vector(&eta_plus, post);
        let post_minus = NaturalParams::from_eta_vector(&eta_minus, post);

        if let (Some(pp), Some(pm)) = (post_plus, post_minus) {
            let f_plus = obs.expected_log_likelihood(&pp);
            let f_minus = obs.expected_log_likelihood(&pm);
            if let Some(g) = grad.get_mut(k) {
                *g = (f_plus - f_minus) / (2.0 * h);
            }
        }
    }

    grad
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use crate::distributions::NaturalParams;
    use crate::graph::{Edge, Graph, Node};
    use crate::observation::Observation;

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

    #[test]
    fn single_node_no_obs() {
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);

        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        let post = &graph.nodes.first().map(|n| &n.post);
        assert!(matches!(post, Some(NaturalParams::Gaussian { .. })));
        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        let eta2 = eta
            .as_ref()
            .and_then(|v| v.get(1).copied())
            .unwrap_or(f64::NAN);
        assert!(eta1.abs() < 1e-10);
        assert!((eta2 + 0.5).abs() < 1e-10);
    }

    #[test]
    fn gaussian_observation_update() {
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 3.0,
                noise_var: 1.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        let post = &graph.nodes.first().map(|n| &n.post);
        assert!(matches!(post, Some(NaturalParams::Gaussian { .. })));
        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        let eta2 = eta
            .as_ref()
            .and_then(|v| v.get(1).copied())
            .unwrap_or(f64::NAN);
        assert!((eta1 - 3.0).abs() < 1e-10, "eta1={eta1}, expected 3.0");
        assert!((eta2 - (-1.0)).abs() < 1e-10, "eta2={eta2}, expected −1.0");
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
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 5.0,
                noise_var: 1.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-8, 1.0);
        assert!(result.converged, "did not converge in 100 iterations");

        let eta_a = graph.nodes.first().map(|n| n.post.eta_vector());
        let a_eta1 = eta_a
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        assert!(a_eta1 > 4.0, "node A eta1={a_eta1} should be > 4");

        let eta_b = graph.nodes.get(1).map(|n| n.post.eta_vector());
        let b_eta1 = eta_b
            .as_ref()
            .and_then(|v| v.get(0).copied())
            .unwrap_or(f64::NAN);
        assert!(
            b_eta1.abs() > 0.01,
            "node B should be influenced by coupling, eta1={b_eta1}"
        );
    }

    #[test]
    fn elbo_increases() {
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            node_a: "A".to_owned(),
            node_b: "B".to_owned(),
            coupling: DMatrix::from_row_slice(2, 2, &[0.05, 0.0, 0.0, 0.0]),
        }];
        let mut graph = Graph::new(nodes, edges);
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 2.0,
                noise_var: 1.0,
            },
        );
        graph.add_observation(
            "B".to_owned(),
            Observation::GaussianNoise {
                value: -1.0,
                noise_var: 2.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 50, 1e-10, 1.0);

        for pair in result.elbo_history.windows(2) {
            if let (Some(prev), Some(next)) = (pair.first(), pair.get(1)) {
                assert!(*next >= *prev - 1e-10, "ELBO decreased: {prev} → {next}",);
            }
        }
    }

    #[test]
    fn bernoulli_observation() {
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
            Observation::NoisyChannel {
                value: true,
                epsilon: 0.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        let eta = graph.nodes.first().map(|n| n.post.eta_vector());
        let eta1 = eta.as_ref().and_then(|v| v.get(0).copied()).unwrap_or(0.0);
        let prob = 1.0 / (1.0 + (-eta1).exp());
        assert!(
            prob > 0.55,
            "p={prob}, should be > 0.55 after observing true"
        );
    }
}
