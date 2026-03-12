//! Tests for inline edges and instrument-based observations.

#[cfg(test)]
mod schema_edge_instrument_tests {
    use crate::schema::validate::parse_yaml;

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
