//! Baillissime application entry point.

use std::io::Write as _;

use tracing_subscriber::EnvFilter;

/// Entry point — sets up tracing and runs the application.
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let greeting = app_core::greet("World");
    writeln!(std::io::stdout(), "{greeting}")?;

    Ok(())
}
