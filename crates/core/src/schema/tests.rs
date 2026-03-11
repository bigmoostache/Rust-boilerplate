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
      type: bernoulli
      p: 0.3
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
      - [0.01]
      - [0.005]
  - node_a: hypertension
    node_b: bmi
    coupling:
      - [0.1, 0.0]
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
    family: { type: bernoulli, p: 0.5 }
    tau: 1.0
edges:
  - node_a: A
    node_b: B
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
                    .any(|e| e.message.contains("expected 2×1"))
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
      type: noisy_channel
      epsilon: 0.1
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
    fn categorical_probs_not_sum_one() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: categorical, probs: [0.3, 0.3, 0.3] }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(err.errors.iter().any(|e| e.message.contains("sum to 1")));
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
  - name: "pois"
    family: { type: poisson, lambda: 5.0 }
    tau: 1.0
  - name: "bern"
    family: { type: bernoulli, p: 0.7 }
    tau: 1.0
  - name: "cat"
    family: { type: categorical, probs: [0.2, 0.3, 0.5] }
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
            assert_eq!(config.graph.num_nodes(), 7);
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
    family: { type: bernoulli, p: 2.0 }
    tau: 0.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: -1.0, delta_t: -1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            // Should have: sigma2, tau, duplicate name, p, tau, tolerance, delta_t
            assert!(
                err.errors.len() >= 5,
                "expected at least 5 errors, got {}",
                err.errors.len()
            );
        }
    }

    #[test]
    fn inline_edges_to() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
    coupled_with:
      - node: B
        coupling:
          - [0.1]
          - [0.0]
  - name: "B"
    family: { type: bernoulli, p: 0.5 }
    tau: 1.0
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_edges(), 1);
            // Edge should be A → B
            assert_eq!(config.graph.neighbors("A").len(), 1);
            assert_eq!(config.graph.neighbors("B").len(), 1);
        }
    }

    #[test]
    fn inline_edges_from() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - name: "B"
    family: { type: bernoulli, p: 0.5 }
    tau: 1.0
    coupled_with:
      - node: A
        coupling:
          - [0.1, 0.0]
edges: []
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_edges(), 1);
            // Edge connects A and B
            assert_eq!(config.graph.neighbors("A").len(), 1);
            assert_eq!(config.graph.neighbors("B").len(), 1);
        }
    }

    #[test]
    fn inline_edges_mixed_with_toplevel() {
        let yaml = r#"
nodes:
  - name: "flu"
    family: { type: bernoulli, p: 0.25 }
    tau: 14.0
    coupled_with:
      - node: headache
        coupling: [[1.5]]
      - node: body_aches
        coupling: [[2.0]]
  - name: "headache"
    family: { type: bernoulli, p: 0.1 }
    tau: 3.0
  - name: "body_aches"
    family: { type: bernoulli, p: 0.05 }
    tau: 3.0
  - name: "sore_throat"
    family: { type: bernoulli, p: 0.05 }
    tau: 5.0
  - name: "tonsillitis"
    family: { type: bernoulli, p: 0.25 }
    tau: 10.0
edges:
  - node_a: tonsillitis
    node_b: sore_throat
    coupling: [[2.5]]
instruments: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            // 2 inline (flu→headache, flu→body_aches) + 1 top-level (tonsillitis→sore_throat)
            assert_eq!(config.graph.num_edges(), 3);
            assert_eq!(config.graph.neighbors("flu").len(), 2);
            assert_eq!(config.graph.neighbors("tonsillitis").len(), 1);
        }
    }

    #[test]
    fn instrument_validation() {
        // Invalid: negative noise_var
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
instruments:
  - name: bad_thermo
    node: A
    model:
      type: gaussian_noise
      noise_var: -1.0
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("noise_var must be > 0"))
            );
        }
    }

    #[test]
    fn duplicate_instrument_name() {
        let yaml = r#"
nodes:
  - name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
instruments:
  - name: thermo
    node: A
    model:
      type: gaussian_noise
      noise_var: 1.0
  - name: thermo
    node: A
    model:
      type: gaussian_noise
      noise_var: 2.0
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_err());
        if let Err(err) = &result {
            assert!(
                err.errors
                    .iter()
                    .any(|e| e.message.contains("duplicate instrument"))
            );
        }
    }

    #[test]
    fn instrument_with_observations() {
        let yaml = r#"
nodes:
  - name: "bp"
    family: { type: gaussian, mu: 120.0, sigma2: 100.0 }
    tau: 30.0
  - name: "has_flu"
    family: { type: bernoulli, p: 0.3 }
    tau: 14.0
edges: []
instruments:
  - name: bp_cuff
    node: bp
    model:
      type: gaussian_noise
      noise_var: 25.0
  - name: symptom_check
    node: has_flu
    model:
      type: noisy_channel
      epsilon: 0.1
observations:
  - instrument: bp_cuff
    value: 145.0
  - instrument: symptom_check
    value: true
inference: { max_iter: 100, tolerance: 0.001, delta_t: 1.0 }
"#;
        let result = parse_yaml(yaml);
        assert!(result.is_ok(), "parse failed: {:?}", result.as_ref().err());
        if let Ok(config) = result {
            assert_eq!(config.graph.num_nodes(), 2);
            assert_eq!(config.graph.observations_for("bp").len(), 1);
            assert_eq!(config.graph.observations_for("has_flu").len(), 1);
        }
    }
}
