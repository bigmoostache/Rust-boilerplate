//! Integration tests for the full inference pipeline.

#[cfg(test)]
mod integration_tests {

    use nalgebra::DMatrix;
    use serde as _;
    use serde_yaml as _;

    use app_core::distributions::NaturalParams;
    use app_core::graph::{Edge, Graph, Node};
    use app_core::inference::coordinate_ascent;
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

    /// Convert a Gaussian observation (value, `noise_var`) into `η_obs`.
    fn gaussian_obs(value: f64, noise_var: f64) -> NaturalParams {
        NaturalParams::Gaussian {
            eta1: value / noise_var,
            eta2: -1.0 / (2.0 * noise_var),
        }
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
    /// Observations (as `η_obs`):
    /// - BP: measured at 145 (noise σ²=25)
    /// - BMI: measured at 30 (noise σ²=4)
    #[test]
    fn three_node_mixed_graph() {
        // Node "BP": Gaussian BP, μ=120, σ²=100
        let bp_params = NaturalParams::Gaussian {
            eta1: 120.0 / 100.0, // μ/σ² = 1.2
            eta2: -1.0 / 200.0,  // -1/(2σ²) = -0.005
        };
        // Node "Hypertension": Beta(0.3, 0.7) → Dirichlet K=2, η = [α−1, β−1] = [−0.7, −0.3]
        let hyp_params = NaturalParams::Dirichlet {
            eta: vec![0.3 - 1.0, 0.7 - 1.0],
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

        // BP → Hypertension coupling: 2×2 matrix (Gaussian d=2, Dirichlet K=2 d=2)
        let bp_hyp = DMatrix::from_row_slice(2, 2, &[0.01, 0.0, 0.0, 0.0]);

        // Hypertension → BMI coupling: 2×2 matrix (Dirichlet K=2 d=2, Gaussian d=2)
        let hyp_bmi = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.0]);

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

        // Add observations as η_obs (NaturalParams)
        graph.add_observation("BP".to_owned(), gaussian_obs(145.0, 25.0));
        graph.add_observation("BMI".to_owned(), gaussian_obs(30.0, 4.0));

        // Run inference
        let result = coordinate_ascent(&mut graph, 5000, 1e-10, 1.0);

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

        // 3. Hypertension α₁ should increase (observed high BP and high BMI)
        if let Some(node) = graph.nodes.get(1) {
            let hyp_eta0 = eta(&node.post, 0);
            let prior_eta0 = 0.3 - 1.0; // = -0.7
            assert!(
                hyp_eta0 > prior_eta0,
                "hypertension η₀ should increase from prior ({prior_eta0}), got {hyp_eta0}"
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

        // 5. Score should be monotonically non-decreasing
        for pair in result.score_history.windows(2) {
            if let (Some(prev), Some(next)) = (pair.first(), pair.get(1)) {
                assert!(*next >= *prev - 1e-8, "Score decreased: {prev} → {next}",);
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
        // η_obs = (2/1, -1/2) = (2, -0.5)
        for _ in 0..5 {
            graph.add_observation("X".to_owned(), gaussian_obs(2.0, 1.0));
        }

        let result = coordinate_ascent(&mut graph, 500, 1e-12, 1.0);
        assert!(
            result.converged,
            "did not converge in {} iters, max_change={}",
            result.iterations, result.max_change
        );

        // Fixed-point formula:
        // η* = (η_relax + 5·η_obs) / (1 + 5 + 1)
        //    = ((0, -0.5) + 5·(2, -0.5)) / 7
        //    = (10, -3) / 7
        //    = (10/7, -3/7)
        if let Some(node) = graph.nodes.first() {
            let eta1 = eta(&node.post, 0);
            let eta2 = eta(&node.post, 1);
            assert!(
                (eta1 - 10.0 / 7.0).abs() < 1e-6,
                "eta1={eta1}, expected 10/7={}",
                10.0 / 7.0
            );
            assert!(
                (eta2 - (-3.0 / 7.0)).abs() < 1e-6,
                "eta2={eta2}, expected -3/7={}",
                -3.0 / 7.0
            );
        }
    }

    /// Test that a Dirichlet node with a conjugate observation shifts correctly.
    #[test]
    fn dirichlet_observation() {
        // Uniform prior over K=3: α = [1, 1, 1] → η = [0, 0, 0]
        let params = NaturalParams::Dirichlet {
            eta: vec![0.0, 0.0, 0.0],
        };
        let mut graph = Graph::new(vec![make_node("Diagnosis", params, 1.0)], vec![]);

        // Observe proportion [0.7, 0.2, 0.1] with κ=10
        // η_obs_k = κ·v_k − 1 = [6, 1, 0]
        graph.add_observation(
            "Diagnosis".to_owned(),
            NaturalParams::Dirichlet {
                eta: vec![6.0, 1.0, 0.0],
            },
        );

        let result = coordinate_ascent(&mut graph, 100, 1e-10, 1.0);
        assert!(result.converged);

        // After observing: η₀ should be largest (class 0 most likely)
        if let Some(node) = graph.nodes.first() {
            let eta0 = eta(&node.post, 0);
            let eta1 = eta(&node.post, 1);
            let eta2 = eta(&node.post, 2);
            assert!(
                eta0 > eta1 && eta1 > eta2,
                "should have η₀ > η₁ > η₂, got [{eta0}, {eta1}, {eta2}]",
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
      type: beta
      alpha: 1.3
      beta: 3.0
    tau: 365.0
edges:
  - node_a: blood_pressure
    node_b: hypertension
    coupling:
      - [0.001, 0.0]
      - [0.0, 0.0]
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
  max_iter: 10000
  tolerance: 1.0e-6
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
            eprintln!(
                "converged={}, iterations={}, max_change={}",
                inf_result.converged, inf_result.iterations, inf_result.max_change
            );
            assert!(
                inf_result.converged,
                "inference did not converge: iterations={}, max_change={}",
                inf_result.iterations, inf_result.max_change
            );

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

        // First inference: observe x=5 with noise σ²=1
        // η_obs = (5, -0.5)
        graph.add_observation("X".to_owned(), gaussian_obs(5.0, 1.0));
        let r1 = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        assert!(r1.converged);

        // Fixed-point: η* = (η_relax + η_obs) / (1 + 1 + 1) = ((0,-0.5)+(5,-0.5))/3 = (5/3, -1/3)
        // Wait — no, that's NOT the old additive formula.
        // η* = (η_relax + η_obs) / (1 + 1 + λ) = (5, -1) / 3 = (5/3, -1/3)
        if let Some(node) = graph.nodes.first() {
            assert!(
                (eta(&node.post, 0) - 5.0 / 3.0).abs() < 1e-6,
                "eta1={}, expected 5/3",
                eta(&node.post, 0)
            );
        }

        // Advance time: copy post→prev, then relax toward epidemio
        advance_and_relax(&mut graph, 2.0_f64.ln()); // decay = 0.5

        // After advance: prev should be old post
        if let Some(node) = graph.nodes.first() {
            let prev_eta1 = eta(&node.prev, 0);
            assert!(
                (prev_eta1 - 5.0 / 3.0).abs() < 1e-6,
                "prev eta1={prev_eta1}, expected 5/3"
            );
            // Relax: (1-0.5)*epidemio + 0.5*prev = 0.5*(0) + 0.5*(5/3) = 5/6
            let relax_eta1 = eta(&node.relax, 0);
            assert!(
                (relax_eta1 - 5.0 / 6.0).abs() < 1e-6,
                "relax eta1={relax_eta1}, expected 5/6"
            );
        }

        // Clear observations and reinfer (no new obs)
        graph.observations.clear();
        let r2 = coordinate_ascent(&mut graph, 100, 1e-12, 1.0);
        assert!(r2.converged);

        // Without observations: η* = η_relax / (1 + 0 + 1) = η_relax / 2
        if let Some(node) = graph.nodes.first() {
            let expected = 5.0 / 6.0 / 2.0; // (5/6) / 2 = 5/12
            assert!(
                (eta(&node.post, 0) - expected).abs() < 1e-6,
                "post eta1={}, expected {}",
                eta(&node.post, 0),
                expected
            );
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
        graph.add_observation("test_node".to_owned(), gaussian_obs(1.0, 1.0));

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
            assert!(yaml_str.contains("score:"));
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
