//! Tests for the YAML schema parsing and validation.

#[cfg(test)]
mod tests {
    use crate::distributions::NaturalParams;
    use crate::schema::validate::parse_yaml;

    const MINIMAL_YAML: &str = r#"
nodes:
  - id: 0
    name: "blood_pressure"
    family:
      type: gaussian
      mu: 120.0
      sigma2: 225.0
    tau: 30.0
edges: []
observations: []
inference:
  max_iter: 100
  tolerance: 1.0e-8
  delta_t: 7.0
"#;

    #[test]
    fn minimal_valid_config() {
        let config = parse_yaml(MINIMAL_YAML).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.graph.num_nodes(), 1);
        assert_eq!(config.graph.num_edges(), 0);
        assert_eq!(config.max_iter, 100);
        assert!((config.tolerance - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn full_three_node_graph() {
        let yaml = r#"
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
  - id: 2
    name: "bmi"
    family:
      type: gaussian
      mu: 25.0
      sigma2: 16.0
    tau: 90.0
edges:
  - from: 0
    to: 1
    coupling:
      - [0.01]
      - [0.005]
  - from: 1
    to: 2
    coupling:
      - [0.1, 0.0]
observations:
  - type: gaussian_noise
    node: 0
    value: 145.0
    noise_var: 25.0
  - type: gaussian_noise
    node: 2
    value: 30.0
    noise_var: 4.0
inference:
  max_iter: 200
  tolerance: 1.0e-10
  delta_t: 7.0
"#;
        let config = parse_yaml(yaml).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.graph.num_nodes(), 3);
        assert_eq!(config.graph.num_edges(), 2);
        assert_eq!(config.graph.observations_for(0).len(), 1);
        assert_eq!(config.graph.observations_for(2).len(), 1);
    }

    #[test]
    fn duplicate_node_id() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - id: 0
    name: "B"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(err.errors.iter().any(|e| e.message.contains("duplicate")));
    }

    #[test]
    fn invalid_sigma2() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: -1.0 }
    tau: 1.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("sigma2 must be > 0"))
        );
    }

    #[test]
    fn invalid_tau() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: -5.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("tau must be > 0"))
        );
    }

    #[test]
    fn unknown_node_in_edge() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges:
  - from: 0
    to: 99
    coupling:
      - [1.0, 0.0]
      - [0.0, 1.0]
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("unknown node ID 99"))
        );
    }

    #[test]
    fn coupling_dimension_mismatch() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - id: 1
    name: "B"
    family: { type: bernoulli, p: 0.5 }
    tau: 1.0
edges:
  - from: 0
    to: 1
    coupling:
      - [1.0, 0.0]
      - [0.0, 1.0]
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("expected 2×1"))
        );
    }

    #[test]
    fn incompatible_observation() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
edges: []
observations:
  - type: bernoulli_exact
    node: 0
    value: true
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("incompatible"))
        );
    }

    #[test]
    fn categorical_probs_not_sum_one() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: categorical, probs: [0.3, 0.3, 0.3] }
    tau: 1.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(err.errors.iter().any(|e| e.message.contains("sum to 1")));
    }

    #[test]
    fn unknown_field_rejected() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
    bogus_field: 42
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        assert!(
            err.errors
                .iter()
                .any(|e| e.message.contains("unknown field"))
        );
    }

    #[test]
    fn canonical_to_natural_roundtrip() {
        // Gaussian: mu=3, sigma2=4 → eta1=0.75, eta2=-0.125
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 3.0, sigma2: 4.0 }
    tau: 1.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let config = parse_yaml(yaml).unwrap_or_else(|e| panic!("{e}"));
        let node = config.graph.node(0).unwrap_or_else(|| panic!("no node 0"));
        let NaturalParams::Gaussian { eta1, eta2 } = node.epidemio else {
            panic!("expected Gaussian");
        };
        assert!((eta1 - 0.75).abs() < 1e-12);
        assert!((eta2 - (-0.125)).abs() < 1e-12);
    }

    #[test]
    fn all_families_parse() {
        let yaml = r#"
nodes:
  - id: 0
    name: "gauss"
    family: { type: gaussian, mu: 0.0, sigma2: 1.0 }
    tau: 1.0
  - id: 1
    name: "gam"
    family: { type: gamma, alpha: 3.0, beta: 2.0 }
    tau: 1.0
  - id: 2
    name: "bet"
    family: { type: beta, alpha: 2.0, beta: 5.0 }
    tau: 1.0
  - id: 3
    name: "pois"
    family: { type: poisson, lambda: 5.0 }
    tau: 1.0
  - id: 4
    name: "bern"
    family: { type: bernoulli, p: 0.7 }
    tau: 1.0
  - id: 5
    name: "cat"
    family: { type: categorical, probs: [0.2, 0.3, 0.5] }
    tau: 1.0
  - id: 6
    name: "dir"
    family: { type: dirichlet, alpha: [2.0, 3.0, 5.0] }
    tau: 1.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: 0.01, delta_t: 1.0 }
"#;
        let config = parse_yaml(yaml).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.graph.num_nodes(), 7);
    }

    #[test]
    fn multiple_errors_collected() {
        let yaml = r#"
nodes:
  - id: 0
    name: "A"
    family: { type: gaussian, mu: 0.0, sigma2: -1.0 }
    tau: -5.0
  - id: 0
    name: "B"
    family: { type: bernoulli, p: 2.0 }
    tau: 0.0
edges: []
observations: []
inference: { max_iter: 10, tolerance: -1.0, delta_t: -1.0 }
"#;
        let err = parse_yaml(yaml).unwrap_err();
        // Should have: sigma2, tau, duplicate id, p, tau, tolerance, delta_t
        assert!(
            err.errors.len() >= 5,
            "expected at least 5 errors, got {}",
            err.errors.len()
        );
    }
}
