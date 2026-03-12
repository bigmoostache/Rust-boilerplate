//! Tests for the calibration linear system.

#[cfg(test)]
mod calibration_tests {
    use crate::calibration::parse::CalibrationConfig;
    use crate::calibration::system::calibrate_edge;
    use crate::schema::raw::GraphConfig;

    /// Parse helper: returns `None` on failure (test will do nothing).
    fn parse_pair(
        graph_yaml: &str,
        cal_yaml: &str,
    ) -> Option<Vec<crate::calibration::parse::ResolvedEdge>> {
        let graph: GraphConfig = serde_yaml::from_str(graph_yaml).ok()?;
        let cal: CalibrationConfig = serde_yaml::from_str(cal_yaml).ok()?;
        crate::calibration::parse::resolve_edges(&cal, &graph).ok()
    }

    /// Test case from Calibration-locale.md:
    /// Gaussian(μ=37, σ²=0.25) ← Beta(α=6, β=34) aka Dirichlet K=2.
    ///
    /// Constraints:
    /// 1. `no_effect` at `x_b = [0.0, 1.0]`  → forces column 0 = 0
    /// 2. `variance_unchanged` at `x_b = [0.99, 0.01]`
    /// 3. `E[temp | flu=0.99] = 39.5`
    ///
    /// Expected: `B = [[0, b₁₂], [0, 0]]`
    /// where `b₁₂ = (39.5 − 37) / (log(0.01) · 0.25)`
    #[test]
    fn gaussian_from_beta_calibration_locale() {
        let graph_yaml = "
nodes:
  - name: temperature
    family:
      type: gaussian
      mu: 37.0
      sigma2: 0.25
    tau: 7.0
  - name: flu
    family:
      type: beta
      alpha: 6.0
      beta: 34.0
    tau: 14.0
";
        let cal_yaml = "
calibration:
  - edge: [temperature, flu]
    statements:
      - type: no_effect
        x_b: [0.0, 1.0]
      - type: variance_unchanged
        x_b: [0.99, 0.01]
      - type: conditional_mean
        x_b: [0.99, 0.01]
        target_mean: 39.5
";
        let edges = parse_pair(graph_yaml, cal_yaml);
        assert!(edges.is_some(), "parse/resolve failed");

        if let Some(edges) = edges
            && let Some(edge) = edges.first()
        {
            let result = calibrate_edge(edge);
            assert!(result.is_ok(), "calibrate failed: {result:?}");

            if let Ok(cal_result) = result {
                // Expected `b12` = (39.5 - 37) / (log(0.01) * 0.25)
                let expected_b12 = (39.5 - 37.0) / (0.01_f64.ln() * 0.25);

                // `B` should be 2×2
                assert_eq!(cal_result.coupling.nrows(), 2);
                assert_eq!(cal_result.coupling.ncols(), 2);

                // Column 0 should be zero (`no_effect` with `log(0) = −∞`)
                let b00 = cal_result.coupling.get((0, 0)).copied().unwrap_or(f64::NAN);
                let b10 = cal_result.coupling.get((1, 0)).copied().unwrap_or(f64::NAN);
                assert!(b00.abs() < 1e-10, "B[0,0] should be 0, got {b00}");
                assert!(b10.abs() < 1e-10, "B[1,0] should be 0, got {b10}");

                // `B[0,1]` should be `b12`
                let b01 = cal_result.coupling.get((0, 1)).copied().unwrap_or(f64::NAN);
                assert!(
                    (b01 - expected_b12).abs() < 1e-6,
                    "B[0,1] = {b01}, expected {expected_b12}"
                );

                // `B[1,1]` should be 0 (from `variance_unchanged`)
                let b11 = cal_result.coupling.get((1, 1)).copied().unwrap_or(f64::NAN);
                assert!(b11.abs() < 1e-10, "B[1,1] should be 0, got {b11}");

                assert!(
                    cal_result.residual < 1e-12,
                    "residual too large: {}",
                    cal_result.residual
                );
            }
        }
    }

    /// Under-determined system: not enough constraints.
    #[test]
    fn under_determined_error() {
        let graph_yaml = "
nodes:
  - name: a
    family:
      type: gaussian
      mu: 0.0
      sigma2: 1.0
    tau: 1.0
  - name: b
    family:
      type: gaussian
      mu: 0.0
      sigma2: 1.0
    tau: 1.0
";
        let cal_yaml = "
calibration:
  - edge: [a, b]
    statements:
      - type: variance_unchanged
        x_b: 1.0
";
        let edges = parse_pair(graph_yaml, cal_yaml);
        assert!(edges.is_some(), "parse/resolve failed");

        if let Some(edges) = edges
            && let Some(edge) = edges.first()
        {
            let err = calibrate_edge(edge);
            assert!(err.is_err(), "expected under-determined error, got Ok");

            if let Err(msg) = err {
                assert!(
                    msg.contains("under-determined"),
                    "expected under-determined error, got: {msg}"
                );
            }
        }
    }

    /// Beta ← Beta (Dirichlet K=2 ← Dirichlet K=2).
    ///
    /// `flu(0.15, s=40)` ← `headache(0.10, s=15)`
    /// When headache = 0.9, flu mean should go to 0.25.
    /// Concentration unchanged + `no_effect` at `[0.0, 1.0]`
    #[test]
    fn beta_from_beta_calibration() {
        let graph_yaml = "
nodes:
  - name: flu
    family:
      type: beta
      alpha: 6.0
      beta: 34.0
    tau: 14.0
  - name: headache
    family:
      type: beta
      alpha: 1.5
      beta: 13.5
    tau: 3.0
";
        let cal_yaml = "
calibration:
  - edge: [flu, headache]
    statements:
      - type: no_effect
        x_b: [0.0, 1.0]
      - type: concentration_unchanged
        x_b: [0.9, 0.1]
      - type: conditional_mean
        x_b: [0.9, 0.1]
        target_mean: 0.25
";
        let edges = parse_pair(graph_yaml, cal_yaml);
        assert!(edges.is_some(), "parse/resolve failed");

        if let Some(edges) = edges
            && let Some(edge) = edges.first()
        {
            let result = calibrate_edge(edge);
            assert!(result.is_ok(), "calibrate failed: {result:?}");

            if let Ok(cal_result) = result {
                // `B` is 2×2.  Column 0 forced to zero by `no_effect`.
                let b00 = cal_result.coupling.get((0, 0)).copied().unwrap_or(f64::NAN);
                let b10 = cal_result.coupling.get((1, 0)).copied().unwrap_or(f64::NAN);
                assert!(b00.abs() < 1e-10, "B[0,0] should be 0, got {b00}");
                assert!(b10.abs() < 1e-10, "B[1,0] should be 0, got {b10}");

                // `B[0,1] = 4 / log(0.1)`
                let expected_b01 = 4.0 / 0.1_f64.ln();
                let b01 = cal_result.coupling.get((0, 1)).copied().unwrap_or(f64::NAN);
                assert!(
                    (b01 - expected_b01).abs() < 1e-6,
                    "B[0,1] = {b01}, expected {expected_b01}"
                );

                // `B[1,1] = −B[0,1]` (concentration unchanged)
                let b11 = cal_result.coupling.get((1, 1)).copied().unwrap_or(f64::NAN);
                assert!(
                    (b11 + b01).abs() < 1e-6,
                    "B[1,1] = {b11}, expected {}",
                    -b01
                );
            }
        }
    }

    /// DOF counting: 2×2 Beta←Beta with `no_effect` + `concentration_unchanged`
    /// + `conditional_mean` gives 3 constraints on 2 free vars → exact.
    #[test]
    fn dof_counting_exact() {
        let graph_yaml = "
nodes:
  - name: a
    family:
      type: beta
      alpha: 3.0
      beta: 7.0
    tau: 1.0
  - name: b
    family:
      type: beta
      alpha: 2.0
      beta: 8.0
    tau: 1.0
";
        let cal_yaml = "
calibration:
  - edge: [a, b]
    statements:
      - type: no_effect
        x_b: [0.0, 1.0]
      - type: concentration_unchanged
        x_b: [0.8, 0.2]
      - type: conditional_mean
        x_b: [0.8, 0.2]
        target_mean: 0.5
";
        let edges = parse_pair(graph_yaml, cal_yaml);
        assert!(edges.is_some(), "parse/resolve failed");

        if let Some(edges) = edges
            && let Some(edge) = edges.first()
        {
            let result = calibrate_edge(edge);
            assert!(result.is_ok(), "calibrate failed: {result:?}");

            if let Ok(cal_result) = result {
                // `no_effect` forces column 0 to 0: 4→2 free vars
                // `concentration_unchanged`: 1 equation
                // `conditional_mean`: 1 equation
                // → 2 free vars, 2 constraints → exact
                assert_eq!(cal_result.n_unknowns, 2);
                assert_eq!(cal_result.n_constraints, 2);
                assert!(
                    cal_result.residual < 1e-12,
                    "residual too large: {}",
                    cal_result.residual
                );

                // Verify column 0 is zero
                let b00 = cal_result.coupling.get((0, 0)).copied().unwrap_or(f64::NAN);
                let b10 = cal_result.coupling.get((1, 0)).copied().unwrap_or(f64::NAN);
                assert!(b00.abs() < 1e-10);
                assert!(b10.abs() < 1e-10);
            }
        }
    }
}
