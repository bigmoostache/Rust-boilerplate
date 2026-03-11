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
    -i, --input <PATH>     Path to the input YAML graph definition (required)
    -o, --output <PATH>    Path to write the output YAML (defaults to stdout)
    -h, --help             Show this help message
";
    let mut stdout = std::io::stdout().lock();
    let _r = stdout.write_all(msg.as_bytes());
}

/// Parsed command-line arguments.
#[derive(Debug)]
enum Command {
    /// Run inference with given input/output paths.
    Infer {
        /// Input YAML path.
        input: String,
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
        let mut input: Option<String> = None;
        let mut output: Option<String> = None;
        let mut i = 0;
        while i < rest.len() {
            let arg = rest.get(i)?;
            match arg.as_str() {
                "-i" | "--input" => {
                    i = i.checked_add(1)?;
                    input = rest.get(i).cloned();
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

        let input = input.or_else(|| {
            let mut stderr = std::io::stderr().lock();
            let _r = writeln!(stderr, "error: --input is required");
            print_usage();
            None
        })?;

        Some(Command::Infer { input, output })
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
        Command::Infer { input, output } => run_infer(&input, output.as_deref()),
    }
}

/// Execute the `infer` subcommand.
fn run_infer(input_path: &str, output_path: Option<&str>) -> anyhow::Result<()> {
    // Read input YAML
    let yaml = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read input file: {input_path}"))?;

    // Parse and validate
    let config =
        app_core::schema::validate::parse_yaml(&yaml).map_err(|errs| anyhow::anyhow!("{errs}"))?;

    // Apply temporal relaxation
    let mut graph = config.graph;
    app_core::temporal::relax_graph(&mut graph, config.delta_t);

    // Run inference
    let result =
        app_core::inference::coordinate_ascent(&mut graph, config.max_iter, config.tolerance);

    // Build and serialize output
    let inference_result = app_core::schema::output::build_result(&graph, &result);
    let output_yaml = app_core::schema::output::to_yaml(&inference_result)
        .context("failed to serialize output")?;

    // Write output
    if let Some(path) = output_path {
        std::fs::write(path, &output_yaml)
            .with_context(|| format!("failed to write output file: {path}"))?;
    } else {
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(output_yaml.as_bytes())
            .context("failed to write to stdout")?;
    }

    Ok(())
}
