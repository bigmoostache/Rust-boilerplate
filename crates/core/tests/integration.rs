//! Integration tests for the full inference pipeline.

use nalgebra::DMatrix;
use serde as _;
use serde_yaml as _;

use app_core::distributions::NaturalParams;
use app_core::graph::{Edge, Graph, Node};
use app_core::inference::coordinate_ascent;
use app_core::observation::Observation;
use app_core::schema::output::{build_result, to_yaml};
use app_core::schema::validate::parse_yaml;
use app_core::temporal::{advance_and_relax, relax_graph};

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

/// Full pipeline: YAML → parse → relax → infer → output YAML → verify.
#[test]
fn yaml_pipeline_roundtrip() {
    let input_yaml = r#"
nodes:
  - id: 0
    name: "blood_pressure"
    family:
      type: gaussian
      mu: 120.0
      sigma2: 225.0
    tau: 30.0
  - id: 1
    name: "hypertension"
    family:
      type: bernoulli
      p: 0.3
    tau: 365.0
edges:
  - from: 0
    to: 1
    coupling:
      - [0.01]
      - [0.005]
observations:
  - type: gaussian_noise
    node: 0
    value: 145.0
    noise_var: 25.0
inference:
  max_iter: 200
  tolerance: 1.0e-10
  delta_t: 7.0
"#;
    // Parse
    let config = parse_yaml(input_yaml).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(config.graph.num_nodes(), 2);
    assert_eq!(config.graph.num_edges(), 1);

    // Relax
    let mut graph = config.graph;
    relax_graph(&mut graph, config.delta_t);

    // Infer
    let result = coordinate_ascent(&mut graph, config.max_iter, config.tolerance);
    assert!(result.converged, "inference did not converge");

    // Build output and serialize to YAML
    let output = build_result(&graph, &result);
    assert!(output.converged);
    assert_eq!(output.posteriors.len(), 2);

    let yaml_str = to_yaml(&output).unwrap_or_else(|e| panic!("serialization failed: {e}"));
    assert!(!yaml_str.is_empty(), "output YAML should not be empty");
    assert!(yaml_str.contains("converged: true"));
    assert!(yaml_str.contains("blood_pressure"));
    assert!(yaml_str.contains("hypertension"));
}

/// Temporal relaxation + inference pipeline: run inference twice
/// with `advance_and_relax` in between to verify state propagation.
#[test]
fn temporal_advance_then_reinfer() {
    let params = NaturalParams::Gaussian {
        eta1: 0.0,
        eta2: -0.5,
    };
    let mut graph = Graph::new(vec![make_node(0, "X", params, 1.0)], vec![]);

    // First inference: observe x=5
    graph.add_observation(
        0,
        Observation::GaussianNoise {
            value: 5.0,
            noise_var: 1.0,
        },
    );
    let r1 = coordinate_ascent(&mut graph, 100, 1e-12);
    assert!(r1.converged);

    // Posterior after first inference: η₁=5, η₂=-1 → μ=2.5, σ²=0.5
    let NaturalParams::Gaussian { eta1: e1_first, .. } = graph.nodes[0].post else {
        panic!("wrong family");
    };
    assert!((e1_first - 5.0).abs() < 1e-10);

    // Advance time: copy post→prev, then relax toward epidemio
    advance_and_relax(&mut graph, 2.0_f64.ln()); // decay = 0.5

    // After advance: prev should be old post
    let NaturalParams::Gaussian { eta1: prev1, .. } = graph.nodes[0].prev else {
        panic!("wrong family");
    };
    assert!((prev1 - 5.0).abs() < 1e-10);

    // Relax: 0.5*epidemio + 0.5*prev = 0.5*0 + 0.5*5 = 2.5
    let NaturalParams::Gaussian { eta1: relax1, .. } = graph.nodes[0].relax else {
        panic!("wrong family");
    };
    assert!((relax1 - 2.5).abs() < 1e-10);

    // Clear observations and reinfer (no new obs)
    graph.observations.clear();
    let r2 = coordinate_ascent(&mut graph, 100, 1e-12);
    assert!(r2.converged);

    // Without observations, posterior = relaxed prior
    let NaturalParams::Gaussian {
        eta1: e1_second, ..
    } = graph.nodes[0].post
    else {
        panic!("wrong family");
    };
    assert!((e1_second - 2.5).abs() < 1e-10);
}

/// Output serialization: verify all expected fields are present.
#[test]
fn output_yaml_structure() {
    let params = NaturalParams::Gaussian {
        eta1: 0.0,
        eta2: -0.5,
    };
    let mut graph = Graph::new(vec![make_node(0, "test_node", params, 1.0)], vec![]);
    graph.add_observation(
        0,
        Observation::GaussianNoise {
            value: 1.0,
            noise_var: 1.0,
        },
    );

    let result = coordinate_ascent(&mut graph, 100, 1e-12);
    let output = build_result(&graph, &result);
    let yaml_str = to_yaml(&output).unwrap_or_else(|e| panic!("{e}"));

    // Verify structural fields
    assert!(yaml_str.contains("converged:"));
    assert!(yaml_str.contains("iterations:"));
    assert!(yaml_str.contains("max_change:"));
    assert!(yaml_str.contains("elbo:"));
    assert!(yaml_str.contains("coupling:"));
    assert!(yaml_str.contains("prior:"));
    assert!(yaml_str.contains("observation:"));
    assert!(yaml_str.contains("entropy:"));
    assert!(yaml_str.contains("posteriors:"));
    assert!(yaml_str.contains("natural_params:"));
    assert!(yaml_str.contains("test_node"));
    assert!(yaml_str.contains("gaussian"));
}
