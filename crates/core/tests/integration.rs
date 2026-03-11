//! Integration tests for the full inference pipeline.

#[cfg(test)]
mod integration_tests {

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

    fn make_node(name: &str, params: NaturalParams, tau: f64) -> Node {
        Node {
            name: name.to_owned(),
            epidemio: params.clone(),
            prev: params.clone(),
            relax: params.clone(),
            post: params,
            tau,
        }
    }

    /// Helper: extract eta components from a `NaturalParams` via `eta_vector`.
    fn eta(params: &NaturalParams, index: usize) -> f64 {
        let v = params.eta_vector();
        v.get(index).copied().unwrap_or(f64::NAN)
    }

    /// A 3-node graph with mixed families:
    /// - Node "BP": Gaussian (systolic BP, μ=120, σ²=100)
    /// - Node "Hypertension": Bernoulli (hypertension diagnosis, p=0.3)
    /// - Node "BMI": Gaussian (BMI, μ=25, σ²=9)
    ///
    /// Couplings:
    /// - BP ↔ Hypertension: BP influences hypertension (2×1 matrix)
    /// - Hypertension ↔ BMI: Hypertension linked to BMI (1×2 matrix)
    ///
    /// Observations:
    /// - BP: measured at 145 (noise σ²=25)
    /// - BMI: measured at 30 (noise σ²=4)
    #[test]
    fn three_node_mixed_graph() {
        // Node "BP": Gaussian BP, μ=120, σ²=100
        let bp_params = NaturalParams::Gaussian {
            eta1: 120.0 / 100.0, // μ/σ² = 1.2
            eta2: -1.0 / 200.0,  // -1/(2σ²) = -0.005
        };
        // Node "Hypertension": Bernoulli, p=0.3 → η = logit(0.3) = ln(3/7)
        let hyp_params = NaturalParams::Bernoulli {
            eta1: (0.3_f64 / 0.7).ln(),
        };
        // Node "BMI": Gaussian BMI, μ=25, σ²=9
        let bmi_params = NaturalParams::Gaussian {
            eta1: 25.0 / 9.0,
            eta2: -1.0 / 18.0,
        };

        let nodes = vec![
            make_node("BP", bp_params, 24.0),
            make_node("Hypertension", hyp_params, 48.0),
            make_node("BMI", bmi_params, 24.0),
        ];

        // BP → Hypertension coupling: 2×1 matrix (suff stats of Gaussian are 2D, Bernoulli is 1D)
        let bp_hyp = DMatrix::from_row_slice(2, 1, &[0.01, 0.0]);

        // Hypertension → BMI coupling: 1×2 matrix
        let hyp_bmi = DMatrix::from_row_slice(1, 2, &[0.5, 0.0]);

        let edges = vec![
            Edge {
                node_a: "BP".to_owned(),
                node_b: "Hypertension".to_owned(),
                coupling: bp_hyp,
            },
            Edge {
                node_a: "Hypertension".to_owned(),
                node_b: "BMI".to_owned(),
                coupling: hyp_bmi,
            },
        ];

        let mut graph = Graph::new(nodes, edges);

        // Add observations
        graph.add_observation(
            "BP".to_owned(),
            Observation::GaussianNoise {
                value: 145.0,
                noise_var: 25.0,
            },
        );
        graph.add_observation(
            "BMI".to_owned(),
            Observation::GaussianNoise {
                value: 30.0,
                noise_var: 4.0,
            },
        );

        // Run inference
        let result = coordinate_ascent(&mut graph, 200, 1e-10, 1.0);

        // 1. Must converge
        assert!(
            result.converged,
            "did not converge in {} iterations, max_change={}",
            result.iterations, result.max_change
        );

        // 2. BP posterior mean should shift toward 145 (observed)
        if let Some(node) = graph.nodes.first() {
            let bp_eta1 = eta(&node.post, 0);
            let bp_eta2 = eta(&node.post, 1);
            let bp_sigma2_post = -1.0 / (2.0 * bp_eta2);
            let bp_mu_post = bp_eta1 * bp_sigma2_post;
            assert!(
                bp_mu_post > 120.0 && bp_mu_post < 145.0,
                "BP posterior mean should be between prior (120) and obs (145), got {bp_mu_post}"
            );
            assert!(
                bp_sigma2_post < 100.0,
                "BP posterior variance should be less than prior (100), got {bp_sigma2_post}"
            );
        }

        // 3. Hypertension probability should increase (observed high BP and high BMI)
        if let Some(node) = graph.nodes.get(1) {
            let hyp_eta = eta(&node.post, 0);
            let prior_logit = (0.3_f64 / 0.7).ln();
            assert!(
                hyp_eta > prior_logit,
                "hypertension logit should increase from prior ({prior_logit}), got {hyp_eta}"
            );
        }

        // 4. BMI posterior mean should shift toward 30 (observed)
        if let Some(node) = graph.nodes.get(2) {
            let bmi_eta1 = eta(&node.post, 0);
            let bmi_eta2 = eta(&node.post, 1);
            let bmi_sigma2_post = -1.0 / (2.0 * bmi_eta2);
            let bmi_mu_post = bmi_eta1 * bmi_sigma2_post;
            assert!(
                bmi_mu_post > 25.0 && bmi_mu_post < 30.0,
                "BMI posterior mean should be between prior (25) and obs (30), got {bmi_mu_post}"
            );
        }

        // 5. ELBO should be monotonically non-decreasing
        for pair in result.elbo_history.windows(2) {
            if let (Some(prev), Some(next)) = (pair.first(), pair.get(1)) {
                assert!(*next >= *prev - 1e-8, "ELBO decreased: {prev} → {next}",);
            }
        }
    }

    /// Test with multiple observations on the same node.
    #[test]
    fn multiple_observations() {
        let params = NaturalParams::Gaussian {
            eta1: 0.0,
            eta2: -0.5,
        };
        let mut graph = Graph::new(vec![make_node("X", params, 1.0)], vec![]);

        // 5 observations at x=2, each with noise σ²=1
        for _ in 0..5 {
            graph.add_observation(
                "X".to_owned(),
                Observation::GaussianNoise {
                    value: 2.0,
                    noise_var: 1.0,
                },
            );
        }

        let result = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        assert!(result.converged);

        // Bayesian update: prior N(0,1), 5 obs at x=2 with σ²_n=1
        // Posterior: η₁ = 0 + 5·2/1 = 10, η₂ = -0.5 + 5·(-0.5) = -3.0
        // → σ²_post = 1/6 ≈ 0.1667, μ_post = 10/(6) = 5/3 ≈ 1.6667
        if let Some(node) = graph.nodes.first() {
            let eta1 = eta(&node.post, 0);
            let eta2 = eta(&node.post, 1);
            assert!((eta1 - 10.0).abs() < 1e-10, "eta1={eta1}, expected 10.0");
            assert!((eta2 - (-3.0)).abs() < 1e-10, "eta2={eta2}, expected -3.0");
        }
    }

    /// Test that a Categorical node with a noisy observation shifts correctly.
    #[test]
    fn categorical_observation() {
        // Uniform prior over K=3 classes
        let params = NaturalParams::Categorical {
            eta: vec![0.0, 0.0],
        };
        let mut graph = Graph::new(vec![make_node("Diagnosis", params, 1.0)], vec![]);

        // Observe class 0 with no confusion (ε=0)
        graph.add_observation(
            "Diagnosis".to_owned(),
            Observation::NoisyCategorical {
                category: 0,
                epsilon: 0.0,
                num_categories: 3,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        // After observing class 0 with ε=0, η₀ should increase (class 0 more likely)
        if let Some(node) = graph.nodes.first() {
            let eta0 = eta(&node.post, 0);
            assert!(
                eta0 > 0.0,
                "η₀ should increase after observing class 0, got {eta0}",
            );
        }
    }

    /// Full pipeline: YAML → parse → relax → infer → output YAML → verify.
    #[test]
    fn yaml_pipeline_roundtrip() {
        let input_yaml = r#"
nodes:
  - name: "blood_pressure"
    family:
      type: gaussian
      mu: 120.0
      sigma2: 225.0
    tau: 30.0
  - name: "hypertension"
    family:
      type: bernoulli
      p: 0.3
    tau: 365.0
edges:
  - node_a: blood_pressure
    node_b: hypertension
    coupling:
      - [0.01]
      - [0.005]
instruments:
  - name: bp_cuff
    node: blood_pressure
    model:
      type: gaussian_noise
      noise_var: 25.0
observations:
  - instrument: bp_cuff
    value: 145.0
inference:
  max_iter: 200
  tolerance: 1.0e-10
  delta_t: 7.0
"#;
        // Parse
        let result = parse_yaml(input_yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_nodes(), 2);
            assert_eq!(config.graph.num_edges(), 1);

            // Relax
            let mut graph = config.graph;
            relax_graph(&mut graph, config.delta_t);

            // Infer
            let inf_result = coordinate_ascent(&mut graph, config.max_iter, config.tolerance, 1.0);
            assert!(inf_result.converged, "inference did not converge");

            // Build output and serialize to YAML
            let output = build_result(&graph, &inf_result);
            assert!(output.converged);
            assert_eq!(output.posteriors.len(), 2);

            let yaml_str = to_yaml(&output);
            assert!(
                yaml_str.is_ok(),
                "serialization failed: {:?}",
                yaml_str.as_ref().err()
            );
            if let Ok(yaml_str) = yaml_str {
                assert!(!yaml_str.is_empty(), "output YAML should not be empty");
                assert!(yaml_str.contains("converged: true"));
                assert!(yaml_str.contains("blood_pressure"));
                assert!(yaml_str.contains("hypertension"));
            }
        }
    }

    /// Temporal relaxation + inference pipeline: run inference twice
    /// with `advance_and_relax` in between to verify state propagation.
    #[test]
    fn temporal_advance_then_reinfer() {
        let params = NaturalParams::Gaussian {
            eta1: 0.0,
            eta2: -0.5,
        };
        let mut graph = Graph::new(vec![make_node("X", params, 1.0)], vec![]);

        // First inference: observe x=5
        graph.add_observation(
            "X".to_owned(),
            Observation::GaussianNoise {
                value: 5.0,
                noise_var: 1.0,
            },
        );
        let r1 = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        assert!(r1.converged);

        // Posterior after first inference: η₁=5, η₂=-1 → μ=2.5, σ²=0.5
        if let Some(node) = graph.nodes.first() {
            assert!((eta(&node.post, 0) - 5.0).abs() < 1e-10);
        }

        // Advance time: copy post→prev, then relax toward epidemio
        advance_and_relax(&mut graph, 2.0_f64.ln()); // decay = 0.5

        // After advance: prev should be old post
        if let Some(node) = graph.nodes.first() {
            assert!((eta(&node.prev, 0) - 5.0).abs() < 1e-10);
            // Relax: 0.5*epidemio + 0.5*prev = 0.5*0 + 0.5*5 = 2.5
            assert!((eta(&node.relax, 0) - 2.5).abs() < 1e-10);
        }

        // Clear observations and reinfer (no new obs)
        graph.observations.clear();
        let r2 = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        assert!(r2.converged);

        // Without observations, posterior = relaxed prior
        if let Some(node) = graph.nodes.first() {
            assert!((eta(&node.post, 0) - 2.5).abs() < 1e-10);
        }
    }

    /// Output serialization: verify all expected fields are present.
    #[test]
    fn output_yaml_structure() {
        let params = NaturalParams::Gaussian {
            eta1: 0.0,
            eta2: -0.5,
        };
        let mut graph = Graph::new(vec![make_node("test_node", params, 1.0)], vec![]);
        graph.add_observation(
            "test_node".to_owned(),
            Observation::GaussianNoise {
                value: 1.0,
                noise_var: 1.0,
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        let output = build_result(&graph, &result);
        let yaml_result = to_yaml(&output);
        assert!(
            yaml_result.is_ok(),
            "serialization failed: {:?}",
            yaml_result.as_ref().err()
        );
        if let Ok(yaml_str) = yaml_result {
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
    }
} // mod integration_tests
