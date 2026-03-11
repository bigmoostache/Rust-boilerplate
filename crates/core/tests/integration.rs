//! Integration tests for the full inference pipeline.

use nalgebra::DMatrix;

use app_core::distributions::NaturalParams;
use app_core::graph::{Edge, Graph, Node};
use app_core::inference::coordinate_ascent;
use app_core::observation::Observation;

fn make_node(id: u32, name: &str, params: NaturalParams, tau: f64) -> Node {
    Node {
        id,
        name: name.to_owned(),
        epidemio: params.clone(),
        prev: params.clone(),
        relax: params.clone(),
        post: params,
        tau,
    }
}

/// A 3-node graph with mixed families:
/// - Node 0: Gaussian (systolic BP, μ=120, σ²=100)
/// - Node 1: Bernoulli (hypertension diagnosis, p=0.3)
/// - Node 2: Gaussian (BMI, μ=25, σ²=9)
///
/// Couplings:
/// - 0 ↔ 1: BP influences hypertension (2×1 matrix)
/// - 1 ↔ 2: Hypertension linked to BMI (1×2 matrix)
///
/// Observations:
/// - Node 0: BP measured at 145 (noise σ²=25)
/// - Node 2: BMI measured at 30 (noise σ²=4)
#[test]
fn three_node_mixed_graph() {
    // Node 0: Gaussian BP, μ=120, σ²=100
    let bp_params = NaturalParams::Gaussian {
        eta1: 120.0 / 100.0, // μ/σ² = 1.2
        eta2: -1.0 / 200.0,  // -1/(2σ²) = -0.005
    };
    // Node 1: Bernoulli hypertension, p=0.3 → η = logit(0.3) = ln(3/7)
    let hyp_params = NaturalParams::Bernoulli {
        eta1: (0.3_f64 / 0.7).ln(),
    };
    // Node 2: Gaussian BMI, μ=25, σ²=9
    let bmi_params = NaturalParams::Gaussian {
        eta1: 25.0 / 9.0,
        eta2: -1.0 / 18.0,
    };

    let nodes = vec![
        make_node(0, "BP", bp_params, 24.0),
        make_node(1, "Hypertension", hyp_params, 48.0),
        make_node(2, "BMI", bmi_params, 24.0),
    ];

    // BP → Hypertension coupling: 2×1 matrix (suff stats of Gaussian are 2D, Bernoulli is 1D)
    // Small positive coupling: higher BP mean → more likely hypertension
    let bp_hyp = DMatrix::from_row_slice(2, 1, &[0.01, 0.0]);

    // Hypertension → BMI coupling: 1×2 matrix
    // Small positive coupling: hypertension → higher BMI expected
    let hyp_bmi = DMatrix::from_row_slice(1, 2, &[0.5, 0.0]);

    let edges = vec![
        Edge {
            i: 0,
            j: 1,
            coupling: bp_hyp,
        },
        Edge {
            i: 1,
            j: 2,
            coupling: hyp_bmi,
        },
    ];

    let mut graph = Graph::new(nodes, edges);

    // Add observations
    graph.add_observation(
        0,
        Observation::GaussianNoise {
            value: 145.0,
            noise_var: 25.0,
        },
    );
    graph.add_observation(
        2,
        Observation::GaussianNoise {
            value: 30.0,
            noise_var: 4.0,
        },
    );

    // Run inference
    let result = coordinate_ascent(&mut graph, 200, 1e-10);

    // 1. Must converge
    assert!(
        result.converged,
        "did not converge in {} iterations, max_change={}",
        result.iterations, result.max_change
    );

    // 2. BP posterior mean should shift toward 145 (observed)
    let NaturalParams::Gaussian { eta1, eta2 } = graph.nodes[0].post else {
        panic!("wrong family for BP");
    };
    let bp_sigma2_post = -1.0 / (2.0 * eta2);
    let bp_mu_post = eta1 * bp_sigma2_post;
    assert!(
        bp_mu_post > 120.0 && bp_mu_post < 145.0,
        "BP posterior mean should be between prior (120) and obs (145), got {bp_mu_post}"
    );
    assert!(
        bp_sigma2_post < 100.0,
        "BP posterior variance should be less than prior (100), got {bp_sigma2_post}"
    );

    // 3. Hypertension probability should increase (observed high BP and high BMI)
    let NaturalParams::Bernoulli { eta1: hyp_eta } = graph.nodes[1].post else {
        panic!("wrong family for Hypertension");
    };
    // Prior logit = ln(3/7) ≈ −0.847. With high BP observation pushing it up,
    // the posterior logit should be higher than the prior.
    let prior_logit = (0.3_f64 / 0.7).ln();
    assert!(
        hyp_eta > prior_logit,
        "hypertension logit should increase from prior ({prior_logit}), got {hyp_eta}"
    );

    // 4. BMI posterior mean should shift toward 30 (observed)
    let NaturalParams::Gaussian {
        eta1: bmi_eta1,
        eta2: bmi_eta2,
    } = graph.nodes[2].post
    else {
        panic!("wrong family for BMI");
    };
    let bmi_sigma2_post = -1.0 / (2.0 * bmi_eta2);
    let bmi_mu_post = bmi_eta1 * bmi_sigma2_post;
    assert!(
        bmi_mu_post > 25.0 && bmi_mu_post < 30.0,
        "BMI posterior mean should be between prior (25) and obs (30), got {bmi_mu_post}"
    );

    // 5. ELBO should be monotonically non-decreasing
    for window in result.elbo_history.windows(2) {
        assert!(
            window[1] >= window[0] - 1e-8,
            "ELBO decreased: {} → {}",
            window[0],
            window[1]
        );
    }
}

/// Test with multiple observations on the same node.
#[test]
fn multiple_observations() {
    let params = NaturalParams::Gaussian {
        eta1: 0.0,
        eta2: -0.5,
    };
    let mut graph = Graph::new(vec![make_node(0, "X", params, 1.0)], vec![]);

    // 5 observations at x=2, each with noise σ²=1
    for _ in 0..5 {
        graph.add_observation(
            0,
            Observation::GaussianNoise {
                value: 2.0,
                noise_var: 1.0,
            },
        );
    }

    let result = coordinate_ascent(&mut graph, 100, 1e-12);
    assert!(result.converged);

    // Bayesian update: prior N(0,1), 5 obs at x=2 with σ²_n=1
    // Posterior: η₁ = 0 + 5·2/1 = 10, η₂ = -0.5 + 5·(-0.5) = -3.0
    // → σ²_post = 1/6 ≈ 0.1667, μ_post = 10/(6) = 5/3 ≈ 1.6667
    let NaturalParams::Gaussian { eta1, eta2 } = graph.nodes[0].post else {
        panic!("wrong family");
    };
    assert!((eta1 - 10.0).abs() < 1e-10, "eta1={eta1}, expected 10.0");
    assert!((eta2 - (-3.0)).abs() < 1e-10, "eta2={eta2}, expected -3.0");
}

/// Test that a Categorical node with an observation shifts correctly.
#[test]
fn categorical_observation() {
    // Uniform prior over K=3 classes
    let params = NaturalParams::Categorical {
        eta: vec![0.0, 0.0],
    };
    let mut graph = Graph::new(vec![make_node(0, "Diagnosis", params, 1.0)], vec![]);

    // Observe class 0
    graph.add_observation(0, Observation::CategoricalExact { category: 0 });

    let result = coordinate_ascent(&mut graph, 100, 1e-10);
    assert!(result.converged);

    let NaturalParams::Categorical { eta } = &graph.nodes[0].post else {
        panic!("wrong family");
    };
    // After observing class 0, η₀ should increase (class 0 more likely)
    assert!(
        eta[0] > 0.0,
        "η₀ should increase after observing class 0, got {}",
        eta[0]
    );
    // η₁ should stay at 0 (no evidence for class 1)
    assert!(
        eta[1].abs() < 1e-10,
        "η₁ should be unchanged, got {}",
        eta[1]
    );
}
