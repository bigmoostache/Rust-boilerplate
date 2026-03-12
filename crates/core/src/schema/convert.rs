//! Validation error types, canonical → natural parameter conversion,
//! and coupling matrix parsing.

use std::fmt;

use nalgebra::DMatrix;

use crate::distributions::NaturalParams;

use super::raw::FamilyDef;

// ---------------------------------------------------------------------------
// Validation error types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Family validation + canonical → natural conversion
// ---------------------------------------------------------------------------

/// Validate canonical params and convert to natural params.
pub(super) fn validate_family(
    family: &FamilyDef,
    path: &str,
) -> Result<NaturalParams, Vec<SchemaError>> {
    let mut errors = Vec::new();

    let params = match family {
        FamilyDef::Gaussian { mu, sigma2 } => {
            if *sigma2 <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.sigma2"),
                    message: format!("sigma2 must be > 0, got {sigma2}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Gaussian {
                eta1: mu / sigma2,
                eta2: -1.0 / (2.0 * sigma2),
            })
        }
        FamilyDef::Gamma { alpha, beta } => {
            if *alpha <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: format!("alpha must be > 0, got {alpha}"),
                });
            }
            if *beta <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.beta"),
                    message: format!("beta must be > 0, got {beta}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Gamma {
                eta1: alpha - 1.0,
                eta2: -beta,
            })
        }
        FamilyDef::Beta { alpha, beta } => {
            // Beta is sugar for Dirichlet with K=2.
            if *alpha <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: format!("alpha must be > 0, got {alpha}"),
                });
            }
            if *beta <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.beta"),
                    message: format!("beta must be > 0, got {beta}"),
                });
            }
            errors.is_empty().then(|| NaturalParams::Dirichlet {
                eta: vec![alpha - 1.0, beta - 1.0],
            })
        }
        FamilyDef::Dirichlet { alpha } => {
            if alpha.len() < 2 {
                errors.push(SchemaError {
                    path: format!("{path}.alpha"),
                    message: "dirichlet needs at least 2 components".to_owned(),
                });
            }
            for (k, ak) in alpha.iter().enumerate() {
                if *ak <= 0.0 {
                    errors.push(SchemaError {
                        path: format!("{path}.alpha[{k}]"),
                        message: format!("alpha must be > 0, got {ak}"),
                    });
                }
            }
            errors.is_empty().then(|| NaturalParams::Dirichlet {
                eta: alpha.iter().map(|a| a - 1.0).collect(),
            })
        }
    };

    if errors.is_empty() {
        Ok(params.unwrap_or_else(|| {
            debug_assert!(false, "params should be Some when no errors");
            NaturalParams::Gaussian {
                eta1: 0.0,
                eta2: -0.5,
            }
        }))
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Coupling matrix parsing
// ---------------------------------------------------------------------------

/// Parse a row-major nested vector into a `DMatrix`.
pub(super) fn parse_coupling(
    rows: &[Vec<f64>],
    path: &str,
) -> Result<DMatrix<f64>, Vec<SchemaError>> {
    if rows.is_empty() {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one row".to_owned(),
        }]);
    }

    let nrows = rows.len();
    let ncols = if let Some(first) = rows.first() {
        first.len()
    } else {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one column".to_owned(),
        }]);
    };

    if ncols == 0 {
        return Err(vec![SchemaError {
            path: path.to_owned(),
            message: "coupling matrix must have at least one column".to_owned(),
        }]);
    }

    let mut errors = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            errors.push(SchemaError {
                path: format!("{path}[{i}]"),
                message: format!(
                    "row has {} columns, expected {ncols} (from first row)",
                    row.len()
                ),
            });
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let data: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}
