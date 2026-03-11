//! Temporal relaxation dynamics.
//!
//! Between successive inference steps the patient's prior "relaxes"
//! back toward the epidemiological reference at a node-specific rate.
//!
//! For a time gap `Δt` and time constant `τ > 0`, the relaxed prior is
//!
//! ```text
//! θ_relax = (1 − e^{−Δt/τ}) · θ_epidemio + e^{−Δt/τ} · θ_prev
//! ```
//!
//! This is a linear interpolation in natural-parameter space where the
//! mixing weight `e^{−Δt/τ}` gives the "memory" of the previous
//! posterior.  When `Δt → 0` we keep the previous posterior;
//! when `Δt → ∞` we revert to the population prior.

use crate::graph::{Graph, Node};

/// Compute the relaxed prior for a single node.
///
/// Sets `node.relax` to the interpolation between `epidemio` (weight
/// `1 − decay`) and `prev` (weight `decay`), where
/// `decay = e^{−Δt / τ}`.
///
/// # Arguments
///
/// * `node` – the node to update (mutated in place).
/// * `delta_t` – elapsed time since the last inference step (must be ≥ 0).
///
/// If `delta_t` is zero the relaxed prior equals `prev` exactly.
/// If `tau` is zero or negative, falls back to `epidemio` (no memory).
pub fn relax_node(node: &mut Node, delta_t: f64) {
    let decay = if node.tau > 0.0 {
        (-delta_t / node.tau).exp()
    } else {
        // τ ≤ 0 means "no temporal memory" → instant revert to epidemio.
        0.0
    };
    // relax = (1 − decay) · epidemio + decay · prev
    // NaturalParams::interpolate does (1 − t) · self + t · other,
    // so we call epidemio.interpolate(&prev, decay).
    node.relax = node.epidemio.interpolate(&node.prev, decay);
}

/// Relax all nodes in the graph with a common time gap `Δt`.
///
/// This should be called **before** running inference so that the
/// relaxed priors are up to date.
pub fn relax_graph(graph: &mut Graph, delta_t: f64) {
    for node in &mut graph.nodes {
        relax_node(node, delta_t);
    }
}

/// Advance time: copy current posteriors into `prev`, then relax.
///
/// This is the full "time step" operation:
/// 1. `prev ← post` for every node (commit the last inference result).
/// 2. Relax all nodes with the given `Δt`.
///
/// After this call, `relax` is ready for a new round of inference.
pub fn advance_and_relax(graph: &mut Graph, delta_t: f64) {
    for node in &mut graph.nodes {
        node.prev = node.post.clone();
    }
    relax_graph(graph, delta_t);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::NaturalParams;

    fn gaussian_node(name: &str, mu: f64, sigma2: f64, tau: f64) -> Node {
        let eta1 = mu / sigma2;
        let eta2 = -1.0 / (2.0 * sigma2);
        let epidemio = NaturalParams::Gaussian { eta1, eta2 };
        Node {
            name: name.to_owned(),
            epidemio: epidemio.clone(),
            prev: epidemio.clone(),
            relax: epidemio.clone(),
            post: epidemio,
            tau,
        }
    }

    /// Helper: extract eta1 from a Gaussian `NaturalParams`.
    fn eta1_of(params: &NaturalParams) -> f64 {
        let v = params.eta_vector();
        v.get(0).copied().unwrap_or(f64::NAN)
    }

    /// Helper: extract (eta1, eta2) from a Gaussian `NaturalParams`.
    fn eta12_of(params: &NaturalParams) -> (f64, f64) {
        let v = params.eta_vector();
        let e1 = v.get(0).copied().unwrap_or(f64::NAN);
        let e2 = v.get(1).copied().unwrap_or(f64::NAN);
        (e1, e2)
    }

    #[test]
    fn zero_delta_t_keeps_prev() {
        let mut node = gaussian_node("A", 5.0, 2.0, 10.0);
        // Set prev to something different from epidemio.
        node.prev = NaturalParams::Gaussian {
            eta1: 1.0,
            eta2: -0.25,
        };
        relax_node(&mut node, 0.0);
        // decay = e^0 = 1 → relax = 0·epidemio + 1·prev = prev
        let (eta1, eta2) = eta12_of(&node.relax);
        assert!((eta1 - 1.0).abs() < 1e-12);
        assert!((eta2 - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn large_delta_t_reverts_to_epidemio() {
        let mut node = gaussian_node("A", 5.0, 2.0, 1.0);
        // Set prev far from epidemio.
        node.prev = NaturalParams::Gaussian {
            eta1: 100.0,
            eta2: -50.0,
        };
        // Δt = 100, τ = 1 → decay ≈ 0 → relax ≈ epidemio
        relax_node(&mut node, 100.0);
        let (eta1, eta2) = eta12_of(&node.relax);
        let (epi1, epi2) = eta12_of(&node.epidemio);
        assert!((eta1 - epi1).abs() < 1e-10);
        assert!((eta2 - epi2).abs() < 1e-10);
    }

    #[test]
    fn intermediate_decay() {
        let mut node = gaussian_node("A", 0.0, 1.0, 1.0);
        // epidemio: μ=0, σ²=1 → η₁=0, η₂=−0.5
        // Set prev: μ=4, σ²=1 → η₁=4, η₂=−0.5
        node.prev = NaturalParams::Gaussian {
            eta1: 4.0,
            eta2: -0.5,
        };
        let delta_t = 2.0_f64.ln(); // decay = e^{-ln2/1} = 0.5
        relax_node(&mut node, delta_t);
        // relax = 0.5 · epidemio + 0.5 · prev
        // η₁ = 0.5 * 0 + 0.5 * 4 = 2
        // η₂ = 0.5 * (-0.5) + 0.5 * (-0.5) = -0.5
        let (eta1, eta2) = eta12_of(&node.relax);
        assert!((eta1 - 2.0).abs() < 1e-12);
        assert!((eta2 - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn zero_tau_reverts_to_epidemio() {
        let mut node = gaussian_node("A", 5.0, 2.0, 0.0); // τ = 0
        node.prev = NaturalParams::Gaussian {
            eta1: 100.0,
            eta2: -50.0,
        };
        relax_node(&mut node, 1.0);
        // τ = 0 → decay = 0 → relax = epidemio
        let (eta1, eta2) = eta12_of(&node.relax);
        let (epi1, epi2) = eta12_of(&node.epidemio);
        assert!((eta1 - epi1).abs() < 1e-12);
        assert!((eta2 - epi2).abs() < 1e-12);
    }

    #[test]
    fn relax_graph_updates_all_nodes() {
        let nodes = vec![
            gaussian_node("A", 0.0, 1.0, 1.0),
            gaussian_node("B", 10.0, 1.0, 2.0),
        ];
        let mut graph = Graph::new(nodes, vec![]);
        // Set prev differently for each.
        if let Some(n) = graph.nodes.get_mut(0) {
            n.prev = NaturalParams::Gaussian {
                eta1: 4.0,
                eta2: -0.5,
            };
        }
        if let Some(n) = graph.nodes.get_mut(1) {
            n.prev = NaturalParams::Gaussian {
                eta1: 20.0,
                eta2: -0.5,
            };
        }
        relax_graph(&mut graph, 2.0_f64.ln());
        // Node 0: τ=1, decay=0.5 → η₁ = 0.5*0 + 0.5*4 = 2
        if let Some(n) = graph.nodes.first() {
            assert!((eta1_of(&n.relax) - 2.0).abs() < 1e-12);
        }
        // Node 1: τ=2, decay=e^{-ln2/2}=e^{-0.347}≈0.707
        if let Some(n) = graph.nodes.get(1) {
            let decay = (-(2.0_f64.ln()) / 2.0).exp();
            let expected = (1.0 - decay).mul_add(10.0, decay * 20.0);
            assert!((eta1_of(&n.relax) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn advance_copies_post_then_relaxes() {
        let nodes = vec![gaussian_node("A", 0.0, 1.0, 1.0)];
        let mut graph = Graph::new(nodes, vec![]);
        // Set a distinct posterior.
        if let Some(n) = graph.nodes.get_mut(0) {
            n.post = NaturalParams::Gaussian {
                eta1: 6.0,
                eta2: -0.5,
            };
        }
        advance_and_relax(&mut graph, 2.0_f64.ln());
        // After advance: prev should be the old post.
        if let Some(n) = graph.nodes.first() {
            assert!((eta1_of(&n.prev) - 6.0).abs() < 1e-12);
            // relax = 0.5*epidemio + 0.5*prev = 0.5*0 + 0.5*6 = 3
            assert!((eta1_of(&n.relax) - 3.0).abs() < 1e-12);
        }
    }
}
