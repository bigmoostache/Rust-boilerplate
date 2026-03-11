//! YAML input types (serde deserialization layer).
//!
//! These types map directly to the YAML structure using canonical
//! (human-readable) parameters.  They are validated and converted
//! to domain types in the `validate` module.

use serde::{Deserialize, Serialize};

/// Top-level YAML document.
///
/// All fields default to empty so that each file can provide a subset
/// of the config.  Multiple files are merged with [`GraphConfig::merge`]
/// before validation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphConfig {
    /// Node definitions (appended across files).
    #[serde(default)]
    pub nodes: Vec<NodeDef>,
    /// Edge definitions (appended across files).
    #[serde(default)]
    pub edges: Vec<EdgeDef>,
    /// Observations (appended across files).
    #[serde(default)]
    pub observations: Vec<ObservationDef>,
    /// Inference settings (last file wins).
    pub inference: Option<InferenceDef>,
}

impl GraphConfig {
    /// Merge another config into this one.
    ///
    /// - `nodes`, `edges`, and `observations` are **appended**.
    /// - `inference` is **overridden** by `other` if `other` provides it.
    pub fn merge(&mut self, other: Self) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
        self.observations.extend(other.observations);
        if other.inference.is_some() {
            self.inference = other.inference;
        }
    }
}

/// A node in the graph.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDef {
    /// Unique node identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Distribution family and canonical parameters.
    pub family: FamilyDef,
    /// Relaxation time constant `τ > 0`.
    pub tau: f64,
}

/// Distribution family with canonical parameters.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum FamilyDef {
    /// `type: gaussian`, `mu`, `sigma2`.
    #[serde(rename = "gaussian")]
    Gaussian {
        /// Mean.
        mu: f64,
        /// Variance (must be > 0).
        sigma2: f64,
    },

    /// `type: gamma`, `alpha`, `beta`.
    #[serde(rename = "gamma")]
    Gamma {
        /// Shape (must be > 0).
        alpha: f64,
        /// Rate (must be > 0).
        beta: f64,
    },

    /// `type: beta`, `alpha`, `beta`.
    #[serde(rename = "beta")]
    Beta {
        /// Shape α (must be > 0).
        alpha: f64,
        /// Shape β (must be > 0).
        beta: f64,
    },

    /// `type: poisson`, `lambda`.
    #[serde(rename = "poisson")]
    Poisson {
        /// Rate (must be > 0).
        lambda: f64,
    },

    /// `type: bernoulli`, `p`.
    #[serde(rename = "bernoulli")]
    Bernoulli {
        /// Probability (must be in `(0, 1)`).
        p: f64,
    },

    /// `type: categorical`, `probs`.
    #[serde(rename = "categorical")]
    Categorical {
        /// Probabilities (must sum to 1, all > 0).
        probs: Vec<f64>,
    },

    /// `type: dirichlet`, `alpha`.
    #[serde(rename = "dirichlet")]
    Dirichlet {
        /// Concentration parameters (all must be > 0).
        alpha: Vec<f64>,
    },
}

/// An edge coupling two nodes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeDef {
    /// Source node ID.
    pub from: u32,
    /// Target node ID.
    pub to: u32,
    /// Coupling matrix as row-major nested vectors.
    pub coupling: Vec<Vec<f64>>,
}

/// An observation attached to a node.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ObservationDef {
    /// Gaussian noise observation.
    #[serde(rename = "gaussian_noise")]
    GaussianNoise {
        /// Node ID.
        node: u32,
        /// Observed value.
        value: f64,
        /// Noise variance (must be > 0).
        noise_var: f64,
    },

    /// Exact categorical observation.
    #[serde(rename = "categorical_exact")]
    CategoricalExact {
        /// Node ID.
        node: u32,
        /// Observed category (0-based).
        category: usize,
    },

    /// Poisson count observation.
    #[serde(rename = "poisson_count")]
    PoissonCount {
        /// Node ID.
        node: u32,
        /// Observed count.
        count: u64,
    },

    /// Bernoulli exact observation.
    #[serde(rename = "bernoulli_exact")]
    BernoulliExact {
        /// Node ID.
        node: u32,
        /// Observed value.
        value: bool,
    },

    /// Beta proportion observation.
    #[serde(rename = "beta_proportion")]
    BetaProportion {
        /// Node ID.
        node: u32,
        /// Observed proportion in `(0, 1)`.
        value: f64,
        /// Concentration `κ > 0`.
        concentration: f64,
    },

    /// Gamma rate observation.
    #[serde(rename = "gamma_rate")]
    GammaRate {
        /// Node ID.
        node: u32,
        /// Observed positive value.
        value: f64,
        /// Shape of the observation noise.
        shape: f64,
    },
}

/// Inference configuration.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceDef {
    /// Maximum coordinate-ascent iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Time since last inference step (for relaxation).
    pub delta_t: f64,
}
