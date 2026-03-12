//! Instrument validation, model–family compatibility, and observation
//! resolution.
//!
//! Instruments define *how* a latent variable is measured (conjugate
//! model + precision parameters).  Observations reference an instrument
//! and carry only the measured value.  This module validates
//! instruments, checks that their model is compatible with the target
//! node's distribution family, and resolves `(instrument, value)` pairs
//! into concrete `NaturalParams` (`η_obs`) values.

use std::collections::HashMap;

use crate::distributions::NaturalParams;
use crate::model::graph::NodeId;

use super::convert::SchemaError;
use super::raw::{FamilyDef, InstrumentDef, ModelDef, ObsValue, ObservationDef};

// ---------------------------------------------------------------------------
// Validated instrument (after parameter checks)
// ---------------------------------------------------------------------------

/// An instrument that has been validated and is ready for observation
/// resolution.
pub(super) struct ValidatedInstrument {
    /// Which node this instrument measures.
    pub node: NodeId,
    /// The validated measurement model.
    pub model: ModelDef,
    /// Number of categories (only set for categorical nodes).
    pub num_categories: Option<usize>,
}

// ---------------------------------------------------------------------------
// Instrument validation
// ---------------------------------------------------------------------------

/// Validate an instrument definition and check model-parameter ranges.
///
/// Does **not** check node existence or model-family compatibility —
/// those are checked separately once we know the node families.
pub(super) fn validate_instrument(
    inst: &InstrumentDef,
    path: &str,
) -> Result<(), Vec<SchemaError>> {
    let mut errors = Vec::new();

    match &inst.model {
        ModelDef::GaussianNoise { noise_var } => {
            if *noise_var <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.noise_var"),
                    message: format!("noise_var must be > 0, got {noise_var}"),
                });
            }
        }
        ModelDef::BetaObs { kappa } | ModelDef::DirichletObs { kappa } => {
            if *kappa <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.kappa"),
                    message: format!("kappa must be > 0, got {kappa}"),
                });
            }
        }
        ModelDef::GammaObs { shape } => {
            if *shape <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.shape"),
                    message: format!("shape must be > 0, got {shape}"),
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Model–family compatibility
// ---------------------------------------------------------------------------

/// Check that an instrument's measurement model is compatible with the
/// target node's distribution family.
pub(super) fn check_model_family_compat(
    model: &ModelDef,
    family: &FamilyDef,
    path: &str,
) -> Option<SchemaError> {
    let compatible = matches!(
        (model, family),
        (ModelDef::GaussianNoise { .. }, FamilyDef::Gaussian { .. })
            | (ModelDef::BetaObs { .. }, FamilyDef::Beta { .. })
            | (ModelDef::BetaObs { .. }, FamilyDef::Dirichlet { .. })
            | (ModelDef::GammaObs { .. }, FamilyDef::Gamma { .. })
            | (ModelDef::DirichletObs { .. }, FamilyDef::Dirichlet { .. })
            | (ModelDef::DirichletObs { .. }, FamilyDef::Beta { .. })
    );

    if compatible {
        None
    } else {
        let model_type = match model {
            ModelDef::GaussianNoise { .. } => "gaussian_noise",
            ModelDef::BetaObs { .. } => "beta_obs",
            ModelDef::GammaObs { .. } => "gamma_obs",
            ModelDef::DirichletObs { .. } => "dirichlet_obs",
        };
        let fam_type = match family {
            FamilyDef::Gaussian { .. } => "gaussian",
            FamilyDef::Gamma { .. } => "gamma",
            FamilyDef::Beta { .. } => "beta",
            FamilyDef::Dirichlet { .. } => "dirichlet",
        };
        Some(SchemaError {
            path: path.to_owned(),
            message: format!(
                "measurement model '{model_type}' is incompatible with node family '{fam_type}'"
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Observation resolution → NaturalParams
// ---------------------------------------------------------------------------

/// Resolve a raw observation (instrument name + value) into a concrete
/// `NaturalParams` (`η_obs`) using the validated instrument map.
///
/// Returns `(node_id, Result<NaturalParams, errors>)`.
pub(super) fn resolve_observation(
    raw: &ObservationDef,
    instruments: &HashMap<String, ValidatedInstrument>,
    path: &str,
) -> (Option<NodeId>, Result<NaturalParams, Vec<SchemaError>>) {
    let Some(inst) = instruments.get(&raw.instrument) else {
        return (
            None,
            Err(vec![SchemaError {
                path: format!("{path}.instrument"),
                message: format!("unknown instrument \"{}\"", raw.instrument),
            }]),
        );
    };

    let node_id = inst.node.clone();
    let result = resolve_value(&inst.model, inst.num_categories, &raw.value, path);
    (Some(node_id), result)
}

/// Convert a raw `ObsValue` into a `NaturalParams` (`η_obs`) given the
/// instrument's model.
fn resolve_value(
    model: &ModelDef,
    _num_categories: Option<usize>,
    value: &ObsValue,
    path: &str,
) -> Result<NaturalParams, Vec<SchemaError>> {
    match model {
        ModelDef::GaussianNoise { noise_var } => {
            let v = extract_float(value, path)?;
            Ok(NaturalParams::Gaussian {
                eta1: v / noise_var,
                eta2: -1.0 / (2.0 * noise_var),
            })
        }

        ModelDef::BetaObs { kappa } => {
            let v = extract_float(value, path)?;
            if v <= 0.0 || v >= 1.0 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be in (0, 1), got {v}"),
                }]);
            }
            // Beta is Dirichlet K=2: η_obs = [κ·v − 1, κ·(1−v) − 1]
            Ok(NaturalParams::Dirichlet {
                eta: vec![kappa * v - 1.0, kappa * (1.0 - v) - 1.0],
            })
        }

        ModelDef::GammaObs { shape } => {
            let v = extract_float(value, path)?;
            if v <= 0.0 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be > 0, got {v}"),
                }]);
            }
            Ok(NaturalParams::Gamma {
                eta1: shape - 1.0,
                eta2: -v,
            })
        }

        ModelDef::DirichletObs { kappa } => {
            let vs = extract_vec(value, path)?;
            if vs.len() < 2 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: "dirichlet observation needs at least 2 values".to_owned(),
                }]);
            }
            for (i, v) in vs.iter().enumerate() {
                if *v <= 0.0 {
                    return Err(vec![SchemaError {
                        path: format!("{path}.value[{i}]"),
                        message: format!("value must be > 0, got {v}"),
                    }]);
                }
            }
            let sum: f64 = vs.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("values must sum to 1, got {sum}"),
                }]);
            }
            Ok(NaturalParams::Dirichlet {
                eta: vs.iter().map(|v| kappa * v - 1.0).collect(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Value extraction helpers
// ---------------------------------------------------------------------------

/// Extract a floating-point value from a raw observation value.
fn extract_float(value: &ObsValue, path: &str) -> Result<f64, Vec<SchemaError>> {
    match value {
        ObsValue::Float(v) => Ok(*v),
        ObsValue::Int(v) => {
            let narrow = i32::try_from(*v).unwrap_or(if *v > 0 { i32::MAX } else { i32::MIN });
            Ok(f64::from(narrow))
        }
        ObsValue::Bool(_) | ObsValue::Vec(_) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: "expected a numeric value".to_owned(),
        }]),
    }
}

/// Extract a vector of floats from a raw observation value.
fn extract_vec(value: &ObsValue, path: &str) -> Result<Vec<f64>, Vec<SchemaError>> {
    match value {
        ObsValue::Vec(vs) => Ok(vs.clone()),
        ObsValue::Bool(_) | ObsValue::Float(_) | ObsValue::Int(_) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: "expected a list of floats".to_owned(),
        }]),
    }
}
