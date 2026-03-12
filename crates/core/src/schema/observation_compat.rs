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

use crate::constants::{EPSILON_ZERO_THRESHOLD, PERFECT_OBS_ETA, POISSON_ZERO_CORRECTION};
use crate::distributions::NaturalParams;
use crate::graph::NodeId;

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
        ModelDef::BernoulliObs { epsilon } => {
            if *epsilon < 0.0 || *epsilon >= 0.5 {
                errors.push(SchemaError {
                    path: format!("{path}.model.epsilon"),
                    message: format!("epsilon must be in [0, 0.5), got {epsilon}"),
                });
            }
        }
        ModelDef::CategoricalObs { epsilon } => {
            if *epsilon < 0.0 || *epsilon >= 1.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.epsilon"),
                    message: format!("epsilon must be in [0, 1), got {epsilon}"),
                });
            }
        }
        ModelDef::PoissonObs { exposure } => {
            if *exposure <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.model.exposure"),
                    message: format!("exposure must be > 0, got {exposure}"),
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
            | (ModelDef::BernoulliObs { .. }, FamilyDef::Bernoulli { .. })
            | (ModelDef::PoissonObs { .. }, FamilyDef::Poisson { .. })
            | (
                ModelDef::CategoricalObs { .. },
                FamilyDef::Categorical { .. }
            )
            | (ModelDef::BetaObs { .. }, FamilyDef::Beta { .. })
            | (ModelDef::GammaObs { .. }, FamilyDef::Gamma { .. })
            | (ModelDef::DirichletObs { .. }, FamilyDef::Dirichlet { .. })
    );

    if compatible {
        None
    } else {
        let model_type = match model {
            ModelDef::GaussianNoise { .. } => "gaussian_noise",
            ModelDef::BernoulliObs { .. } => "bernoulli_obs",
            ModelDef::PoissonObs { .. } => "poisson_obs",
            ModelDef::CategoricalObs { .. } => "categorical_obs",
            ModelDef::BetaObs { .. } => "beta_obs",
            ModelDef::GammaObs { .. } => "gamma_obs",
            ModelDef::DirichletObs { .. } => "dirichlet_obs",
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
    num_categories: Option<usize>,
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

        ModelDef::BernoulliObs { epsilon } => {
            let v = extract_bool(value, path)?;
            // ε = 0 → perfect observation → η_obs = ±∞ → clamp
            let eta = if *epsilon < EPSILON_ZERO_THRESHOLD {
                if v { PERFECT_OBS_ETA } else { -PERFECT_OBS_ETA }
            } else {
                let log_odds = ((1.0 - epsilon) / epsilon).ln();
                if v { log_odds } else { -log_odds }
            };
            Ok(NaturalParams::Bernoulli { eta1: eta })
        }

        ModelDef::PoissonObs { exposure } => {
            let count = extract_nonneg_int(value, path)?;
            // Continuity correction for count=0
            let effective_count = if count == 0 {
                POISSON_ZERO_CORRECTION
            } else {
                f64::from(u32::try_from(count).unwrap_or(u32::MAX))
            };
            Ok(NaturalParams::Poisson {
                eta1: (effective_count / exposure).ln(),
            })
        }

        ModelDef::CategoricalObs { epsilon } => {
            let cat = extract_nonneg_int(value, path)?;
            let category = usize::try_from(cat).unwrap_or(usize::MAX);
            let k = num_categories.unwrap_or(0);
            if k > 0 && category >= k {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!(
                        "category index {category} out of range for {k}-category node"
                    ),
                }]);
            }
            // Categorical natural params are K-1 log-ratios (vs reference class K-1).
            let dim = if k > 1 { k.saturating_sub(1) } else { 1 };

            // Compute log-odds from confusion matrix:
            // P(measure cat | truly cat) = 1 − ε
            // P(measure cat | truly j≠cat) = ε / (K−1)
            // log-odds = ln((1−ε) / (ε/(K−1))) = ln((1−ε)(K−1) / ε)
            let log_odds = if *epsilon < EPSILON_ZERO_THRESHOLD {
                PERFECT_OBS_ETA // perfect observation → clamp
            } else {
                let k_f = f64::from(u32::try_from(k.max(2)).unwrap_or(u32::MAX));
                ((1.0 - epsilon) * (k_f - 1.0) / epsilon).ln()
            };

            let mut eta = vec![0.0; dim];
            if category < dim
                && let Some(slot) = eta.get_mut(category)
            {
                *slot = log_odds;
            }
            // If category == reference class (last), all log-ratios get
            // −log_odds (reference is more likely than each alternative).
            if category >= dim {
                for e in &mut eta {
                    *e = -log_odds;
                }
            }
            Ok(NaturalParams::Categorical { eta })
        }

        ModelDef::BetaObs { kappa } => {
            let v = extract_float(value, path)?;
            if v <= 0.0 || v >= 1.0 {
                return Err(vec![SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be in (0, 1), got {v}"),
                }]);
            }
            Ok(NaturalParams::Beta {
                eta1: kappa * v - 1.0,
                eta2: kappa * (1.0 - v) - 1.0,
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
