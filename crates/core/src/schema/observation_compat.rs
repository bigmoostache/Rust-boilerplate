//! Instrument validation, model–family compatibility, and observation
//! resolution.
//!
//! Instruments define *how* a latent variable is measured (noise model +
//! parameters).  Observations reference an instrument and carry only
//! the measured value.  This module validates instruments, checks that
//! their measurement model is compatible with the target node's
//! distribution family, and resolves `(instrument, value)` pairs into
//! concrete [`Observation`] values.

use std::collections::HashMap;

use crate::graph::NodeId;
use crate::observation::Observation;

use super::diag::SchemaError;
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
        ModelDef::NoisyChannel { epsilon } => {
            if *epsilon < 0.0 || *epsilon >= 0.5 {
                errors.push(SchemaError {
                    path: format!("{path}.model.epsilon"),
                    message: format!("epsilon must be in [0, 0.5), got {epsilon}"),
                });
            }
        }
        ModelDef::PoissonCount => {} // no parameters to validate
        ModelDef::NoisyCategorical { epsilon } => {
            if *epsilon < 0.0 || *epsilon >= 1.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.epsilon"),
                    message: format!("epsilon must be in [0, 1), got {epsilon}"),
                });
            }
        }
        ModelDef::BetaConcentration { kappa } | ModelDef::DirichletConcentration { kappa } => {
            if *kappa <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.kappa"),
                    message: format!("kappa must be > 0, got {kappa}"),
                });
            }
        }
        ModelDef::GammaRate { shape } => {
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
            | (ModelDef::NoisyChannel { .. }, FamilyDef::Bernoulli { .. })
            | (ModelDef::PoissonCount, FamilyDef::Poisson { .. })
            | (
                ModelDef::NoisyCategorical { .. },
                FamilyDef::Categorical { .. }
            )
            | (ModelDef::BetaConcentration { .. }, FamilyDef::Beta { .. })
            | (ModelDef::GammaRate { .. }, FamilyDef::Gamma { .. })
            | (
                ModelDef::DirichletConcentration { .. },
                FamilyDef::Dirichlet { .. }
            )
    );

    if compatible {
        None
    } else {
        let model_type = match model {
            ModelDef::GaussianNoise { .. } => "gaussian_noise",
            ModelDef::NoisyChannel { .. } => "noisy_channel",
            ModelDef::PoissonCount => "poisson_count",
            ModelDef::NoisyCategorical { .. } => "noisy_categorical",
            ModelDef::BetaConcentration { .. } => "beta_concentration",
            ModelDef::GammaRate { .. } => "gamma_rate",
            ModelDef::DirichletConcentration { .. } => "dirichlet_concentration",
        };
        let fam_type = match family {
            FamilyDef::Gaussian { .. } => "gaussian",
            FamilyDef::Gamma { .. } => "gamma",
            FamilyDef::Beta { .. } => "beta",
            FamilyDef::Poisson { .. } => "poisson",
            FamilyDef::Bernoulli { .. } => "bernoulli",
            FamilyDef::Categorical { .. } => "categorical",
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
// Observation resolution
// ---------------------------------------------------------------------------

/// Resolve a raw observation (instrument name + value) into a concrete
/// [`Observation`] using the validated instrument map.
///
/// Returns `(node_id, Result<Observation, errors>)`.
pub(super) fn resolve_observation(
    raw: &ObservationDef,
    instruments: &HashMap<String, ValidatedInstrument>,
    path: &str,
) -> (Option<NodeId>, Result<Observation, Vec<SchemaError>>) {
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

/// Convert a raw `ObsValue` into a concrete `Observation` given the
/// instrument's model.
fn resolve_value(
    model: &ModelDef,
    num_categories: Option<usize>,
    value: &ObsValue,
    path: &str,
) -> Result<Observation, Vec<SchemaError>> {
    match model {
        ModelDef::GaussianNoise { noise_var } => {
            let v = extract_float(value, path)?;
            Ok(Observation::GaussianNoise {
                value: v,
                noise_var: *noise_var,
            })
        }

        ModelDef::NoisyChannel { epsilon } => {
            let v = extract_bool(value, path)?;
            Ok(Observation::NoisyChannel {
                value: v,
                epsilon: *epsilon,
            })
        }

        ModelDef::PoissonCount => {
            let v = extract_nonneg_int(value, path)?;
            Ok(Observation::PoissonCount { count: v })
        }

        ModelDef::NoisyCategorical { epsilon } => {
            let v = extract_nonneg_int(value, path)?;
            let category = usize::try_from(v).unwrap_or(usize::MAX);
            let k = num_categories.unwrap_or(0);
            if k > 0 && category >= k {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!(
                        "category index {category} out of range for {k}-category node"
                    ),
                }]);
            }
            Ok(Observation::NoisyCategorical {
                category,
                epsilon: *epsilon,
                num_categories: k,
            })
        }

        ModelDef::BetaConcentration { kappa } => {
            let v = extract_float(value, path)?;
            if v <= 0.0 || v >= 1.0 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be in (0, 1), got {v}"),
                }]);
            }
            Ok(Observation::BetaConcentration {
                value: v,
                kappa: *kappa,
            })
        }

        ModelDef::GammaRate { shape } => {
            let v = extract_float(value, path)?;
            if v <= 0.0 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be > 0, got {v}"),
                }]);
            }
            Ok(Observation::GammaRate {
                value: v,
                shape: *shape,
            })
        }

        ModelDef::DirichletConcentration { kappa } => {
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
            Ok(Observation::DirichletConcentration {
                values: vs,
                kappa: *kappa,
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
            // Lossless: clamp to i32 range then use From<i32> for f64.
            let narrow = i32::try_from(*v).unwrap_or(if *v > 0 { i32::MAX } else { i32::MIN });
            Ok(f64::from(narrow))
        }
        ObsValue::Bool(_) | ObsValue::Vec(_) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: "expected a numeric value".to_owned(),
        }]),
    }
}

/// Extract a boolean value from a raw observation value.
fn extract_bool(value: &ObsValue, path: &str) -> Result<bool, Vec<SchemaError>> {
    match value {
        ObsValue::Bool(v) => Ok(*v),
        ObsValue::Float(_) | ObsValue::Int(_) | ObsValue::Vec(_) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: "expected a boolean value (true/false)".to_owned(),
        }]),
    }
}

/// Extract a non-negative integer from a raw observation value.
fn extract_nonneg_int(value: &ObsValue, path: &str) -> Result<u64, Vec<SchemaError>> {
    match value {
        ObsValue::Int(v) if *v >= 0 => Ok(u64::try_from(*v).unwrap_or(u64::MAX)),
        ObsValue::Int(v) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: format!("expected a non-negative integer, got {v}"),
        }]),
        ObsValue::Bool(_) | ObsValue::Float(_) | ObsValue::Vec(_) => Err(vec![SchemaError {
            path: format!("{path}.value"),
            message: "expected an integer value".to_owned(),
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
