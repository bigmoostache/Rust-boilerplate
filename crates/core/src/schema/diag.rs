//! Validation error types for YAML schema parsing.

use std::fmt;

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct SchemaError {
    /// Dot-path to the offending field (e.g. `"nodes[0].params.sigma2"`).
    pub path: String,
    /// Human-readable description of the problem.
    pub message: String,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Collection of all validation errors found during parsing.
#[derive(Debug, Clone)]
pub struct SchemaErrors {
    /// All errors found.
    pub errors: Vec<SchemaError>,
}

impl fmt::Display for SchemaErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "YAML validation failed with {} error(s):",
            self.errors.len()
        )?;
        for err in &self.errors {
            writeln!(f, "  • {err}")?;
        }
        Ok(())
    }
}
