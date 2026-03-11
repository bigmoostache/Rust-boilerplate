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
pub fn coordinate_ascent(graph: &mut Graph, max_iter: usize, tol: f64) -> ConvergenceResult {
    let mut elbo_history = Vec::with_capacity(max_iter);

    for iter in 0..max_iter {
        let mut max_change = 0.0_f64;

        for node_idx in 0..graph.num_nodes() {
            let new_eta = compute_optimal_eta(graph, node_idx);
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

        let elbo = graph.elbo();
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

    let elbo = graph.elbo();
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
/// ```text
/// η_i^{new} = η_i^{relax} + Σ_{j} coupling_contribution_j + obs_contribution
/// ```
fn compute_optimal_eta(graph: &Graph, node_idx: usize) -> DVector<f64> {
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

        // Bernoulli node + Bernoulli observation: exact
        (NaturalParams::Bernoulli { .. }, Observation::BernoulliExact { value, count }) => {
            // ln p(obs|x) = obs·ln(x) + (1−obs)·ln(1−x), repeated count times
            // The η contribution is count · obs (0 or count)
            let v = if *value { f64::from(*count) } else { 0.0 };
            DVector::from_vec(vec![v])
        }

        // Poisson node + Poisson count: exact conjugate
        (NaturalParams::Poisson { eta1 }, Observation::PoissonCount { count }) => {
            let k = f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
            DVector::from_vec(vec![k - eta1.exp()])
        }

        // Categorical node + categorical observation: exact
        (NaturalParams::Categorical { eta }, Observation::CategoricalExact { category }) => {
            let dim = eta.len();
            let mut grad = DVector::zeros(dim);
            if let Some(entry) = grad.get_mut(*category) {
                *entry = 1.0;
            }
            grad
        }

        // For non-conjugate combos: numerical gradient via finite differences
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
        // With no observations and no couplings, posterior should stay at relax
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        let result = coordinate_ascent(&mut graph, 100, 1e-10);

        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        // Posterior should equal relax (unchanged)
        let NaturalParams::Gaussian { eta1, eta2 } = graph.nodes[0].post else {
            panic!("wrong family");
        };
        assert!(eta1.abs() < 1e-10);
        assert!((eta2 + 0.5).abs() < 1e-10); // −1/(2·1) = −0.5
    }

    #[test]
    fn gaussian_observation_update() {
        // Prior: N(0, 1), Observation: x=3 with noise σ²_n=1
        // Bayesian update: posterior = N(1.5, 0.5)
        // η₁_post = 0/1 + 3/1 = 3, η₂_post = −1/2 + (−1/2) = −1
        // → μ = −η₁/(2η₂) = −3/(−2) = 1.5, σ² = −1/(2η₂) = 0.5 ✓
        let mut graph = Graph::new(vec![make_gaussian_node("A", 0.0, 1.0)], vec![]);
        graph.add_observation(
            "A".to_owned(),
            Observation::GaussianNoise {
                value: 3.0,
                noise_var: 1.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10);
        assert!(result.converged);

        let NaturalParams::Gaussian { eta1, eta2 } = graph.nodes[0].post else {
            panic!("wrong family");
        };
        // η₁ = 0 + 3/1 = 3
        assert!((eta1 - 3.0).abs() < 1e-10, "eta1={eta1}, expected 3.0");
        // η₂ = −0.5 + (−0.5) = −1.0
        assert!((eta2 - (-1.0)).abs() < 1e-10, "eta2={eta2}, expected −1.0");
    }

    #[test]
    fn two_coupled_gaussians() {
        // Two Gaussian nodes coupled with a small matrix.
        // Observation on node A should propagate to node B.
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        // Small coupling so it converges quickly
        let edges = vec![Edge {
            i: "A".to_owned(),
            j: "B".to_owned(),
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

        let result = coordinate_ascent(&mut graph, 100, 1e-8);
        assert!(result.converged, "did not converge in 100 iterations");

        // Node A should have posterior shifted toward 5.0
        let NaturalParams::Gaussian { eta1: a_eta1, .. } = graph.nodes[0].post else {
            panic!("wrong family");
        };
        // η₁ of node A should be roughly 5 (from obs) + coupling contribution
        assert!(a_eta1 > 4.0, "node A eta1={a_eta1} should be > 4");

        // Node B should have been pulled by coupling
        let NaturalParams::Gaussian { eta1: b_eta1, .. } = graph.nodes[1].post else {
            panic!("wrong family");
        };
        // Node B has no observation, so it's pulled by coupling with node A
        assert!(
            b_eta1.abs() > 0.01,
            "node B should be influenced by coupling, eta1={b_eta1}"
        );
    }

    #[test]
    fn elbo_increases() {
        // ELBO should be non-decreasing across iterations
        let nodes = vec![
            make_gaussian_node("A", 0.0, 1.0),
            make_gaussian_node("B", 0.0, 1.0),
        ];
        let edges = vec![Edge {
            i: "A".to_owned(),
            j: "B".to_owned(),
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

        let result = coordinate_ascent(&mut graph, 50, 1e-10);

        // Check ELBO is non-decreasing (with tolerance for floating point)
        for window in result.elbo_history.windows(2) {
            assert!(
                window[1] >= window[0] - 1e-10,
                "ELBO decreased: {} → {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn bernoulli_observation() {
        // Bernoulli node with uniform prior (η₁ = 0 → p = 0.5)
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
            Observation::BernoulliExact {
                value: true,
                count: 1,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10);
        assert!(result.converged);

        // After observing true, η₁ should be 0 + 1 = 1
        // p = sigmoid(1) ≈ 0.731
        let NaturalParams::Bernoulli { eta1 } = graph.nodes[0].post else {
            panic!("wrong family");
        };
        assert!((eta1 - 1.0).abs() < 1e-10, "eta1={eta1}, expected 1.0");
    }
}
