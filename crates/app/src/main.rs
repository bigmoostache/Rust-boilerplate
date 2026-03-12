//! Baillissime CLI — virtual patient inference engine.
//!
//! Usage:
//!   `app infer -i <input.yaml> [-o <output.yaml>]`
//!
//! If no output path is given, results are written to stdout.

use std::io::Write as _;

use anyhow::Context as _;
use tracing_subscriber::EnvFilter;

/// Print usage information and exit.
fn print_usage() {
    let msg = "\
Baillissime — virtual patient model CLI

USAGE:
    app <COMMAND> [OPTIONS]

COMMANDS:
    infer    Run inference on a patient graph

INFER OPTIONS:
    -i, --input <PATH>         Path to a YAML input file (repeatable, merged in order)
    -o, --output <PATH>        Path to write the output YAML (defaults to stdout)
    --entropy_scale <FLOAT>    Entropy scaling factor λ (default: 1.0, from YAML if set)
    -h, --help                 Show this help message

EXAMPLES:
    app infer -i graph.yaml                              # single file
    app infer -i nodes.yaml -i edges.yaml -i patient.yaml  # merged files
    app infer -i nodes.yaml -i patient.yaml -o result.yaml  # with output file
";
    let mut stdout = std::io::stdout().lock();
    let _r = stdout.write_all(msg.as_bytes());
}

/// Parsed command-line arguments.
#[derive(Debug)]
enum Command {
    /// Run inference with given input/output paths.
    Infer {
        /// Input YAML paths (merged in order).
        inputs: Vec<String>,
        /// Optional output YAML path.
        output: Option<String>,
        /// Entropy scaling factor (overrides YAML if provided).
        entropy_scale: Option<f64>,
    },
}

/// Parse command-line arguments manually.
///
/// Returns `None` if help was requested or arguments are invalid.
fn parse_args() -> Option<Command> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_usage();
        return None;
    }

    let subcommand = args.first()?;

    if subcommand == "infer" {
        let rest = args.get(1..).unwrap_or_default();
        let mut inputs: Vec<String> = Vec::new();
        let mut output: Option<String> = None;
        let mut entropy_scale: Option<f64> = None;
        let mut i = 0;
        while i < rest.len() {
            let arg = rest.get(i)?;
            match arg.as_str() {
                "-i" | "--input" => {
                    i = i.checked_add(1)?;
                    if let Some(path) = rest.get(i) {
                        inputs.push(path.clone());
                    }
                }
                "-o" | "--output" => {
                    i = i.checked_add(1)?;
                    output = rest.get(i).cloned();
                }
                "--entropy_scale" => {
                    i = i.checked_add(1)?;
                    if let Some(val) = rest.get(i) {
                        entropy_scale = val.parse().ok();
                    }
                }
                "-h" | "--help" => {
                    print_usage();
                    return None;
                }
                other => {
                    let mut stderr = std::io::stderr().lock();
                    let _r = writeln!(stderr, "unknown option: {other}");
                    print_usage();
                    return None;
                }
            }
            i = i.checked_add(1)?;
        }

        if inputs.is_empty() {
            let mut stderr = std::io::stderr().lock();
            let _r = writeln!(stderr, "error: at least one --input is required");
            print_usage();
            return None;
        }

        Some(Command::Infer {
            inputs,
            output,
            entropy_scale,
        })
    } else if subcommand == "-h" || subcommand == "--help" {
        print_usage();
        None
    } else {
        let mut stderr = std::io::stderr().lock();
        let _r = writeln!(stderr, "unknown command: {subcommand}");
        print_usage();
        None
    }
}

/// Entry point — parses CLI and dispatches to subcommands.
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let Some(command) = parse_args() else {
        return Ok(());
    };

    match command {
        Command::Infer {
            inputs,
            output,
            entropy_scale,
        } => run_infer(&inputs, output.as_deref(), entropy_scale),
    }
}

/// Execute the `infer` subcommand.
fn run_infer(
    input_paths: &[String],
    output_path: Option<&str>,
    cli_entropy_scale: Option<f64>,
) -> anyhow::Result<()> {
    // Read all input YAMLs
    let mut yamls: Vec<String> = Vec::with_capacity(input_paths.len());
    for path in input_paths {
        let yaml = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read input file: {path}"))?;
        yamls.push(yaml);
    }

    // Parse and validate (merge multiple files)
    let yaml_refs: Vec<&str> = yamls.iter().map(String::as_str).collect();
    let config = app_core::schema::validate::parse_yamls(&yaml_refs)
        .map_err(|errs| anyhow::anyhow!("{errs}"))?;

    // CLI flag overrides YAML setting
    let entropy_scale = cli_entropy_scale.unwrap_or(config.entropy_scale);

    // Keep resolved observations for display (before moving config.graph)
    let resolved_observations = config.resolved_observations;

    // Apply temporal relaxation
    let mut graph = config.graph;
    app_core::temporal::relax_graph(&mut graph, config.delta_t);

    // Run inference
    let result = app_core::inference::coordinate_ascent(
        &mut graph,
        config.max_iter,
        config.tolerance,
        entropy_scale,
    );

    // Build output
    let inference_result = app_core::schema::output::build_result(&graph, &result);

    // Write full YAML to file if requested
    if let Some(path) = output_path {
        let output_yaml = app_core::schema::output::to_yaml(&inference_result)
            .context("failed to serialize output")?;
        std::fs::write(path, &output_yaml)
            .with_context(|| format!("failed to write output file: {path}"))?;
    }

    // Always print human-readable summary to stdout
    let mut stdout = std::io::stdout().lock();

    // Graph dimensions
    let n_nodes = graph.nodes.len();
    let n_edges = graph.edges.len();
    let n_obs: usize = graph.observations.values().map(Vec::len).sum();
    let _r1 = writeln!(
        stdout,
        "Computed on {n_nodes} nodes, {n_edges} edges and {n_obs} observations."
    );

    // Parameter count by family
    let mut n_coupling: usize = 0;
    for edge in &graph.edges {
        n_coupling =
            n_coupling.saturating_add(edge.coupling.nrows().saturating_mul(edge.coupling.ncols()));
    }
    let mut n_epidemio: usize = 0;
    let mut n_tau: usize = 0;
    for node in &graph.nodes {
        n_epidemio = n_epidemio.saturating_add(node.epidemio.suff_stat_dim());
        n_tau = n_tau.saturating_add(1);
    }
    // Inference config: max_iter, tolerance, entropy_scale, delta_t = 4
    let n_inference: usize = 4;
    let n_total = n_coupling
        .saturating_add(n_epidemio)
        .saturating_add(n_tau)
        .saturating_add(n_inference);
    let _r_params = writeln!(
        stdout,
        "\nModel parameters to configure:\n  \
         coupling (B_ij entries) : {n_coupling:>4}\n  \
         priors (epidemio η)     : {n_epidemio:>4}\n  \
         relaxation (τ per node) : {n_tau:>4}\n  \
         inference config        : {n_inference:>4}\n  \
         ────────────────────────────────\n  \
         total                   : {n_total:>4}"
    );

    if inference_result.converged {
        let _r2 = writeln!(
            stdout,
            "Converged in {} iterations.\n",
            inference_result.iterations
        );
    } else {
        let _r2 = writeln!(
            stdout,
            "Did NOT converge after {} iterations.\n",
            inference_result.iterations
        );
    }

    // Group resolved observations by node for display
    let mut obs_by_node: std::collections::HashMap<
        &str,
        Vec<&app_core::schema::output::ResolvedObservation>,
    > = std::collections::HashMap::new();
    for obs in &resolved_observations {
        obs_by_node.entry(obs.node.as_str()).or_default().push(obs);
    }

    // Build table rows: (name, family, param1, param2)
    let rows: Vec<(String, String, String, String)> = inference_result
        .posteriors
        .iter()
        .map(posterior_columns)
        .collect();

    // Compute column widths
    let w_name = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let w_family = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
    let w_p1 = rows.iter().map(|r| r.2.len()).max().unwrap_or(0);

    for (name, family, p1, p2) in &rows {
        if p2.is_empty() {
            let _r = writeln!(
                stdout,
                "  {name:<w_name$}  {family:<w_family$}  {p1:>w_p1$}"
            );
        } else {
            let _r = writeln!(
                stdout,
                "  {name:<w_name$}  {family:<w_family$}  {p1:>w_p1$}  {p2}"
            );
        }
        // Show observations for this node in purple
        if let Some(obs_list) = obs_by_node.get(name.as_str()) {
            for obs in obs_list {
                let obs_posterior = app_core::schema::output::NodePosterior {
                    name: obs.instrument.clone(),
                    family: obs.eta_obs.to_canonical(),
                    natural_params: obs.eta_obs.eta_vector().as_slice().to_vec(),
                };
                let (_, obs_family, obs_p1, obs_p2) = posterior_columns(&obs_posterior);
                if obs_p2.is_empty() {
                    let _r = writeln!(
                        stdout,
                        "  \x1b[35m{:<w_name$}  {obs_family:<w_family$}  {obs_p1:>w_p1$}\x1b[0m",
                        obs.instrument,
                    );
                } else {
                    let _r = writeln!(
                        stdout,
                        "  \x1b[35m{:<w_name$}  {obs_family:<w_family$}  {obs_p1:>w_p1$}  {obs_p2}\x1b[0m",
                        obs.instrument,
                    );
                }
            }
        }
    }

    Ok(())
}

/// Extract table columns from a node posterior:
/// `(name, family, canonical_params, interpretable_summary)`.
fn posterior_columns(
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
            // Var(Beta) = α·β / ((α+β)² · (α+β+1))
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
