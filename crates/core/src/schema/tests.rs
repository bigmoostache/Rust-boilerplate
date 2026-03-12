//! Tests for the YAML schema parsing and validation.

#[cfg(test)]
mod schema_tests {
    use crate::distributions::NaturalParams;
    use crate::schema::validate::parse_yaml;

    const MINIMAL_YAML: &str = r#"
nodes:
  - name: "blood_pressure"
    family:
      type: gaussian
      mu: 120.0
      sigma2: 225.0
    tau: 30.0
edges: []
instruments: []
observations: []
inference:
  max_iter: 100
  tolerance: 1.0e-8
  delta_t: 7.0
"#;

    #[test]
    fn minimal_valid_config() {
        let result = parse_yaml(MINIMAL_YAML);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_nodes(), 1);
            assert_eq!(config.graph.num_edges(), 0);
            assert_eq!(config.max_iter, 100);
            assert!((config.tolerance - 1e-8).abs() < 1e-15);
        }
    }

    #[test]
    fn full_three_node_graph() {
        let yaml = r#"
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
  - name: "bmi"
    family:
      type: gaussian
      mu: 25.0
      sigma2: 16.0
    tau: 90.0
edges:
  - node_a: blood_pressure
    node_b: hypertension
    coupling:
      - [0.01, 0.0]
      - [0.0, 0.005]
  - node_a: hypertension
    node_b: bmi
    coupling:
      - [0.1, 0.0]
      - [0.0, 0.0]
instruments:
  - name: bp_cuff
    node: blood_pressure
    model:
      type: gaussian_noise
      noise_var: 25.0
  - name: bmi_scale
    node: bmi
    model:
      type: gaussian_noise
      noise_var: 4.0
observations:
  - instrument: bp_cuff
    value: 145.0
  - instrument: bmi_scale
    value: 30.0
inference:
  max_iter: 200
  tolerance: 1.0e-10
  delta_t: 7.0
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_nodes(), 3);
            assert_eq!(config.graph.num_edges(), 2);
            assert_eq!(config.graph.observations_for("blood_pressure").len(), 1);
            assert_eq!(config.graph.observations_for("bmi").len(), 1);
        }
    }

    #[test]
    fn duplicate_node_name() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(err.errors.iter().any(|e| e.message.contains("duplicate")));
        }
    }

    #[test]
    fn invalid_sigma2() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: -1.0 }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("sigma2 must be > 0"))
            );
        }
    }

    #[test]
    fn invalid_tau() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: -5.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("tau must be > 0"))
            );
        }
    }

    #[test]
    fn unknown_node_in_edge() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges:
  - node_a: A
    node_b: nonexistent
    coupling:
      - [1.0, 0.0]
      - [0.0, 1.0]
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("unknown node"))
            );
        }
    }

    #[test]
    fn coupling_dimension_mismatch() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - name: "B"
    family: { type: beta, alpha: 1.0, beta: 1.0 }
    tau: 1.0
edges:
  - node_a: A
    node_b: B
    coupling:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("expected 2×2")),
                "expected dimension mismatch error, got: {:?}",
                err.errors
            );
        }
    }

    #[test]
    fn incompatible_instrument_model() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
instruments:
  - name: wrong_instrument
    node: A
    model:
      type: beta_obs
      kappa: 10.0
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("incompatible"))
            );
        }
    }

    #[test]
    fn unknown_instrument_in_observation() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
instruments: []
observations:
  - instrument: nonexistent
    value: 42.0
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("unknown instrument"))
            );
        }
    }

    #[test]
    fn dirichlet_invalid_alpha() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: dirichlet, alpha: [2.0, -1.0, 3.0] }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("alpha must be > 0"))
            );
        }
    }

    #[test]
    fn unknown_field_rejected() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
    bogus_field: 42
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("unknown field"))
            );
        }
    }

    #[test]
    fn canonical_to_natural_roundtrip() {
        // Gaussian: mu=3, sigma2=4 → eta1=0.75, eta2=-0.125
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 3.0, sigma2: 4.0 }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            if let Some(node) = config.graph.node("A") {
                assert!(
                    matches!(node.epidemio, NaturalParams::Gaussian { .. }),
                    "expected Gaussian"
                );
                let v = node.epidemio.eta_vector();
                let eta1 = v.get(0).copied().unwrap_or(f64::NAN);
                let eta2 = v.get(1).copied().unwrap_or(f64::NAN);
                assert!((eta1 - 0.75).abs() < 1e-12);
                assert!((eta2 - (-0.125)).abs() < 1e-12);
            } else {
                assert!(config.graph.node("A").is_some(), "no node A");
            }
        }
    }

    #[test]
    fn all_families_parse() {
        let yaml = r#"
nodes:
  - name: "gauss"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - name: "gam"
    family: { type: gamma, alpha: 3.0, beta: 2.0 }
    tau: 1.0
  - name: "bet"
    family: { type: beta, alpha: 2.0, beta: 5.0 }
    tau: 1.0
  - name: "dir"
    family: { type: dirichlet, alpha: [2.0, 3.0, 5.0] }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_nodes(), 4);
        }
    }

    #[test]
    fn multiple_errors_collected() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: -1.0 }
    tau: -5.0
  - name: "A"
    family: { type: beta, alpha: -1.0, beta: 0.5 }
    tau: 0.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: -1.0, delta_t: -1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            // Should have: sigma2, tau, duplicate name, alpha, tau, tolerance, delta_t
            assert!(
                err.errors.len() >= 5,
                "expected at least 5 errors, got {}: {:?}",
                err.errors.len(),
                err.errors
            );
        }
    }
}
