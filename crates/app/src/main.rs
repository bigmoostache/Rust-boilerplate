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
    -i, --input <PATH>     Path to a YAML input file (repeatable, merged in order)
    -o, --output <PATH>    Path to write the output YAML (defaults to stdout)
    -h, --help             Show this help message

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

        Some(Command::Infer { inputs, output })
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
        Command::Infer { inputs, output } => run_infer(&inputs, output.as_deref()),
    }
}

/// Execute the `infer` subcommand.
fn run_infer(input_paths: &[String], output_path: Option<&str>) -> anyhow::Result<()> {
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

    // Apply temporal relaxation
    let mut graph = config.graph;
    app_core::temporal::relax_graph(&mut graph, config.delta_t);

    // Run inference
    let result =
        app_core::inference::coordinate_ascent(&mut graph, config.max_iter, config.tolerance);

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
    if inference_result.converged {
        let _r = writeln!(
            stdout,
            "Converged in {} iterations.\n",
            inference_result.iterations
        );
    } else {
        let _r = writeln!(
            stdout,
            "Did NOT converge after {} iterations.\n",
            inference_result.iterations
        );
    }
    for posterior in &inference_result.posteriors {
        let _r = writeln!(stdout, "- {}", format_posterior(posterior));
    }

    Ok(())
}

/// Format a node posterior as a short human-readable string.
fn format_posterior(p: &app_core::schema::output::NodePosterior) -> String {
    match &p.family {
        app_core::schema::raw::FamilyDef::Gaussian { mu, sigma2 } => {
            format!("{} (gaussian) mu={mu:.2}, sigma2={sigma2:.2}", p.name)
        }
        app_core::schema::raw::FamilyDef::Bernoulli { p: prob } => {
            format!("{} (bernoulli) p={prob:.2}", p.name)
        }
        app_core::schema::raw::FamilyDef::Gamma { alpha, beta } => {
            format!("{} (gamma) alpha={alpha:.2}, beta={beta:.2}", p.name)
        }
        app_core::schema::raw::FamilyDef::Beta { alpha, beta } => {
            format!("{} (beta) alpha={alpha:.2}, beta={beta:.2}", p.name)
        }
        app_core::schema::raw::FamilyDef::Poisson { lambda } => {
            format!("{} (poisson) lambda={lambda:.2}", p.name)
        }
        app_core::schema::raw::FamilyDef::Categorical { probs } => {
            let ps: Vec<String> = probs.iter().map(|v| format!("{v:.2}")).collect();
            format!("{} (categorical) probs=[{}]", p.name, ps.join(", "))
        }
        app_core::schema::raw::FamilyDef::Dirichlet { alpha } => {
            let as_str: Vec<String> = alpha.iter().map(|v| format!("{v:.2}")).collect();
            format!("{} (dirichlet) alpha=[{}]", p.name, as_str.join(", "))
        }
    }
}
