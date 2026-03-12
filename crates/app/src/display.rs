//! Display helpers for CLI output formatting.

use std::io::Write;

/// Extract table columns from a node posterior:
/// `(name, family, canonical_params, interpretable_summary)`.
pub(crate) fn posterior_columns(
    p: &app_core::schema::output::NodePosterior,
) -> (String, String, String, String) {
    match &p.family {
        app_core::schema::raw::FamilyDef::Gaussian { mu, sigma2 } => {
            let std = sigma2.sqrt();
            (
                p.name.clone(),
                "gaussian".to_owned(),
                format!("mu={mu:.2}  σ²={sigma2:.2}"),
                format!("{mu:.2} ± {std:.2}"),
            )
        }
        app_core::schema::raw::FamilyDef::Gamma { alpha, beta } => {
            let mean = alpha / beta;
            let std = alpha.sqrt() / beta;
            (
                p.name.clone(),
                "gamma".to_owned(),
                format!("α={alpha:.2}  β={beta:.2}"),
                format!("mean={mean:.2} ± {std:.2}"),
            )
        }
        app_core::schema::raw::FamilyDef::Beta { alpha, beta } => {
            let sum = alpha + beta;
            let mean = alpha / sum;
            let denom = sum.powi(2) * (sum + 1.0);
            let std = (alpha * beta / denom).sqrt();
            (
                p.name.clone(),
                "beta".to_owned(),
                format!("α={alpha:.2}  β={beta:.2}"),
                format!("mean={mean:.2} ± {std:.2}"),
            )
        }
        app_core::schema::raw::FamilyDef::Dirichlet { alpha } => {
            let sum: f64 = alpha.iter().sum();
            let probs: Vec<String> = alpha.iter().map(|a| format!("{:.2}", a / sum)).collect();
            let alphas: Vec<String> = alpha.iter().map(|v| format!("{v:.2}")).collect();
            (
                p.name.clone(),
                "dirichlet".to_owned(),
                format!("α=[{}]", alphas.join(", ")),
                format!("probs=[{}]", probs.join(", ")),
            )
        }
    }
}

/// Print a calibrated edge result: matrix + YAML snippet.
pub(crate) fn print_calibrated_edge(
    out: &mut impl Write,
    edge: &app_core::calibration::parse::ResolvedEdge,
    result: &app_core::calibration::system::CalibratedEdge,
) {
    drop(writeln!(
        out,
        "  unknowns: {}  constraints: {}  residual: {:.2e}",
        result.n_unknowns, result.n_constraints, result.residual
    ));
    drop(writeln!(out, "  B ="));
    for row in 0..result.coupling.nrows() {
        let vals: Vec<String> = (0..result.coupling.ncols())
            .map(|col| {
                format!(
                    "{:>10.6}",
                    result.coupling.get((row, col)).copied().unwrap_or(f64::NAN)
                )
            })
            .collect();
        drop(writeln!(out, "    [{}]", vals.join(", ")));
    }

    // Print YAML snippet for copy-paste into nodes.yaml
    drop(writeln!(
        out,
        "\n  YAML snippet (add to {} node):",
        edge.name_a
    ));
    drop(writeln!(out, "    coupled_with:"));
    drop(writeln!(out, "      - node: {}", edge.name_b));
    drop(writeln!(out, "        coupling:"));
    for row in 0..result.coupling.nrows() {
        let vals: Vec<String> = (0..result.coupling.ncols())
            .map(|col| {
                format!(
                    "{}",
                    result.coupling.get((row, col)).copied().unwrap_or(f64::NAN)
                )
            })
            .collect();
        drop(writeln!(out, "          - [{}]", vals.join(", ")));
    }
}
