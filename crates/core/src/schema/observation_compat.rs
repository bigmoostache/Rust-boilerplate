//! Observation validation and family-compatibility checking.

use crate::graph::NodeId;
use crate::observation::Observation;

use super::diag::SchemaError;
use super::raw::{FamilyDef, ObservationDef};

// ---------------------------------------------------------------------------
// Observation validation
// ---------------------------------------------------------------------------

/// Validate a raw observation and return (`node_id`, parsed observation).
pub(super) fn validate_observation(
    raw: &ObservationDef,
    path: &str,
) -> (NodeId, Result<Observation, Vec<SchemaError>>) {
    match raw {
        ObservationDef::GaussianNoise {
            node,
            value,
            noise_var,
        } => {
            if *noise_var <= 0.0 {
                (
                    *node,
                    Err(vec![SchemaError {
                        path: format!("{path}.noise_var"),
                        message: format!("noise_var must be > 0, got {noise_var}"),
                    }]),
                )
            } else {
                (
                    *node,
                    Ok(Observation::GaussianNoise {
                        value: *value,
                        noise_var: *noise_var,
                    }),
                )
            }
        }
        ObservationDef::CategoricalExact { node, category } => (
            *node,
            Ok(Observation::CategoricalExact {
                category: *category,
            }),
        ),
        ObservationDef::PoissonCount { node, count } => {
            (*node, Ok(Observation::PoissonCount { count: *count }))
        }
        ObservationDef::BernoulliExact { node, value } => {
            (*node, Ok(Observation::BernoulliExact { value: *value }))
        }
        ObservationDef::BetaProportion {
            node,
            value,
            concentration,
        } => {
            let mut errors = Vec::new();
            if *value <= 0.0 || *value >= 1.0 {
                errors.push(SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be in (0, 1), got {value}"),
                });
            }
            if *concentration <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.concentration"),
                    message: format!("concentration must be > 0, got {concentration}"),
                });
            }
            if errors.is_empty() {
                (
                    *node,
                    Ok(Observation::BetaProportion {
                        value: *value,
                        concentration: *concentration,
                    }),
                )
            } else {
                (*node, Err(errors))
            }
        }
        ObservationDef::GammaRate { node, value, shape } => {
            let mut errors = Vec::new();
            if *value <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.value"),
                    message: format!("value must be > 0, got {value}"),
                });
            }
            if *shape <= 0.0 {
                errors.push(SchemaError {
                    path: format!("{path}.shape"),
                    message: format!("shape must be > 0, got {shape}"),
                });
            }
            if errors.is_empty() {
                (
                    *node,
                    Ok(Observation::GammaRate {
                        value: *value,
                        shape: *shape,
                    }),
                )
            } else {
                (*node, Err(errors))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Family–observation compatibility
// ---------------------------------------------------------------------------

/// Check that an observation type is compatible with a node's family.
pub(super) fn check_obs_family_compat(
    obs: &Observation,
    family: &FamilyDef,
    path: &str,
) -> Option<SchemaError> {
    let compatible = matches!(
        (obs, family),
        (
            Observation::GaussianNoise { .. },
            FamilyDef::Gaussian { .. }
        ) | (
            Observation::CategoricalExact { .. },
            FamilyDef::Categorical { .. }
        ) | (Observation::PoissonCount { .. }, FamilyDef::Poisson { .. })
            | (
                Observation::BernoulliExact { .. },
                FamilyDef::Bernoulli { .. }
            )
            | (Observation::BetaProportion { .. }, FamilyDef::Beta { .. })
            | (Observation::GammaRate { .. }, FamilyDef::Gamma { .. })
    );

    if compatible {
        None
    } else {
        let obs_type = match obs {
            Observation::GaussianNoise { .. } => "gaussian_noise",
            Observation::CategoricalExact { .. } => "categorical_exact",
            Observation::PoissonCount { .. } => "poisson_count",
            Observation::BernoulliExact { .. } => "bernoulli_exact",
            Observation::BetaProportion { .. } => "beta_proportion",
            Observation::GammaRate { .. } => "gamma_rate",
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
                "observation type '{obs_type}' is incompatible with node family '{fam_type}'"
            ),
        })
    }
}
