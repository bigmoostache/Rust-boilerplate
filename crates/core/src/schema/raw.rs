//! YAML input types (serde deserialization layer).
//!
//! These types map directly to the YAML structure using canonical
//! (human-readable) parameters.  They are validated and converted
//! to domain types in the `validate` module.
//!
//! ## Instrument-based observation model
//!
//! Instruments are defined once with their measurement mode and noise
//! parameters.  Observations reference an instrument and provide only
//! the measured value.  The instrument carries all the noise semantics.

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
    /// Instrument definitions (appended across files).
    #[serde(default)]
    pub instruments: Vec<InstrumentDef>,
    /// Observations (appended across files).
    #[serde(default)]
    pub observations: Vec<ObservationDef>,
    /// Inference settings (last file wins).
    pub inference: Option<InferenceDef>,
}

impl GraphConfig {
    /// Merge another config into this one.
    ///
    /// - `nodes`, `edges`, `instruments`, and `observations` are **appended**.
    /// - `inference` is **overridden** by `other` if `other` provides it.
    pub fn merge(&mut self, other: Self) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
        self.instruments.extend(other.instruments);
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
    /// Human-readable name (unique identifier).
    pub name: String,
    /// Distribution family and canonical parameters.
    pub family: FamilyDef,
    /// Relaxation time constant `τ > 0`.
    pub tau: f64,
    /// Inline edges: this node is coupled with the listed nodes.
    #[serde(default)]
    pub coupled_with: Vec<InlineEdge>,
}

/// An inline edge declared on a node definition.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InlineEdge {
    /// The other node's name.
    pub node: String,
    /// Coupling matrix as row-major nested vectors.
    pub coupling: Vec<Vec<f64>>,
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

/// An instrument definition — a reusable measurement device.
///
/// Each instrument is bound to a specific node and defines the
/// measurement mode (how the observation relates to the latent
/// variable) plus the noise parameters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstrumentDef {
    /// Unique instrument name (referenced by observations).
    pub name: String,
    /// Node this instrument measures.
    pub node: String,
    /// Measurement model and noise parameters.
    pub model: ModelDef,
}

/// Measurement model with noise parameters.
///
/// Each variant defines how the measured value relates to the latent
/// variable, including the instrument's noise characteristics.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ModelDef {
    /// Gaussian noise: `p(obs | x) = N(obs; x, noise_var)`.
    #[serde(rename = "gaussian_noise")]
    GaussianNoise {
        /// Noise variance `σ² > 0`.
        noise_var: f64,
    },

    /// Noisy binary channel: `P(obs=1 | p) = (1−ε)·p + ε·(1−p)`.
    #[serde(rename = "noisy_channel")]
    NoisyChannel {
        /// Symmetric error probability `ε ∈ [0, 0.5)`.
        epsilon: f64,
    },

    /// Direct Poisson count (no noise parameters needed).
    #[serde(rename = "poisson_count")]
    PoissonCount,

    /// Noisy categorical: uniform confusion with probability `ε`.
    #[serde(rename = "noisy_categorical")]
    NoisyCategorical {
        /// Confusion probability `ε ∈ [0, 1)`.
        epsilon: f64,
    },

    /// Beta concentration: observed proportion ~ `Beta(κp, κ(1−p))`.
    #[serde(rename = "beta_concentration")]
    BetaConcentration {
        /// Concentration `κ > 0` (higher = more precise).
        kappa: f64,
    },

    /// Gamma rate: observed value ~ `Gamma(shape, mean_rate)`.
    #[serde(rename = "gamma_rate")]
    GammaRate {
        /// Shape of the observation noise.
        shape: f64,
    },

    /// Dirichlet concentration: observed proportions ~ `Dir(κ·p)`.
    #[serde(rename = "dirichlet_concentration")]
    DirichletConcentration {
        /// Concentration `κ > 0`.
        kappa: f64,
    },
}

/// An edge coupling two nodes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeDef {
    /// First node name.
    pub node_a: String,
    /// Second node name.
    pub node_b: String,
    /// Coupling matrix as row-major nested vectors.
    pub coupling: Vec<Vec<f64>>,
}

/// An observation event — a measured value from an instrument.
///
/// The observation references an instrument by name; the instrument
/// defines which node is measured and the noise model.  The
/// observation only provides the measured value.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationDef {
    /// Instrument name (must match a defined instrument).
    pub instrument: String,
    /// Measured value.
    ///
    /// The type of value depends on the instrument's model:
    /// - `gaussian_noise`: a float (e.g. `39.5`)
    /// - `noisy_channel`: a boolean (`true` / `false`)
    /// - `poisson_count`: an integer (e.g. `7`)
    /// - `noisy_categorical`: an integer category index (e.g. `2`)
    /// - `beta_concentration`: a float proportion in (0, 1)
    /// - `gamma_rate`: a positive float
    /// - `dirichlet_concentration`: a list of floats summing to ~1
    pub value: ObsValue,
}

/// The measured value of an observation.
///
/// Untagged so YAML can use bare values: `value: 39.5`, `value: true`,
/// `value: [0.3, 0.7]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ObsValue {
    /// A boolean value (for noisy channel).
    Bool(bool),
    /// An integer value (for poisson count or categorical index).
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A vector of floats (for dirichlet concentration).
    Vec(Vec<f64>),
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
