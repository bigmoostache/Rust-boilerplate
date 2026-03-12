//! Exponential family distributions in natural parameterization.
//!
//! Each variant of [`NaturalParams`] represents a distribution from an
//! exponential family, stored in its natural (canonical) parameterization.
//! All key quantities — expected sufficient statistics, entropy,
//! cross-entropy, log-partition — are computed in closed form.

mod dirichlet;
pub(crate) mod gamma;
mod gaussian;

use nalgebra::{DMatrix, DVector};

use crate::schema::raw::FamilyDef;

use dirichlet::DirichletDist;
use gamma::GammaDist;
use gaussian::Gaussian;

// ---------------------------------------------------------------------------
// Exponential family trait — generic formulas for entropy, cross-entropy.
// ---------------------------------------------------------------------------

/// Trait for exponential family distributions.
///
/// An exponential family density has the form:
///
/// ```text
/// p(x; η) = h(x) · exp(η · T(x) − A(η))
/// ```
///
/// Implementors provide the three family-specific ingredients;
/// generic functions [`ef_entropy`] and [`ef_cross_entropy`] derive
/// the rest.
pub(crate) trait ExponentialFamily {
    /// Log-partition function `A(η)`.
    fn log_partition(eta: &DVector<f64>) -> f64;

    /// Expected sufficient statistics `E_η[T(x)] = ∇A(η)`.
    fn expected_suff_stats(eta: &DVector<f64>) -> DVector<f64>;

    /// Fisher information matrix `F(η) = ∇²A(η) = Cov_η[T(x)]`.
    fn fisher_information(eta: &DVector<f64>) -> DMatrix<f64>;

    /// Project `η` onto the valid domain for this family (in place).
    ///
    /// After a damped update, natural parameters may leave the valid
    /// domain (e.g. Gaussian η₂ ≥ 0, Gamma η₁ ≤ −1).  This clamps
    /// each component to a small margin inside the boundary, using
    /// [`NATURAL_PARAM_EPS`](crate::model::constants::NATURAL_PARAM_EPS).
    fn project(eta: &mut DVector<f64>);

    /// Expected log-base-measure `E_η[ln h(x)]`.
    ///
    /// Defaults to `0` — correct for families where `h(x) = 1`.
    fn expected_log_base_measure(_eta: &DVector<f64>) -> f64 {
        0.0
    }
}

/// Differential entropy `H(η) = −E_η[ln p_η(x)]`.
///
/// ```text
/// H = −E[ln h(x)] − η · E[T(x)] + A(η)
/// ```
fn ef_entropy<F: ExponentialFamily>(eta: &DVector<f64>) -> f64 {
    -F::expected_log_base_measure(eta) - eta.dot(&F::expected_suff_stats(eta))
        + F::log_partition(eta)
}

/// Cross-entropy `E_me[ln p_other(x)]`.
///
/// ```text
/// E_me[ln p_other] = E_me[ln h(x)] + η_other · E_me[T(x)] − A(η_other)
/// ```
fn ef_cross_entropy<F: ExponentialFamily>(me: &DVector<f64>, other: &DVector<f64>) -> f64 {
    F::expected_log_base_measure(me) + other.dot(&F::expected_suff_stats(me))
        - F::log_partition(other)
}

/// Natural parameters for the seven supported exponential families.
///
/// The natural parameterization `η` is chosen so that the density writes
/// `p(x) = h(x) exp(η·T(x) − A(η))` where `T` is the sufficient
/// statistic vector and `A` the log-partition function.
#[derive(Debug, Clone)]
pub enum NaturalParams {
    /// Gaussian with natural parameters `(η₁, η₂)`.
    ///
    /// - `η₁ = μ / σ²`
    /// - `η₂ = −1 / (2 σ²)`  (must be negative)
    /// - Sufficient statistics: `T(x) = (x, x²)`, dimension `d = 2`
    Gaussian {
        /// `η₁ = μ / σ²`
        eta1: f64,
        /// `η₂ = −1 / (2σ²)` — must be strictly negative.
        eta2: f64,
    },

    /// Gamma with natural parameters `(η₁, η₂)`.
    ///
    /// - `η₁ = α − 1`  (must be > −1)
    /// - `η₂ = −β`     (must be negative)
    /// - Sufficient statistics: `T(x) = (ln x, x)`, dimension `d = 2`
    Gamma {
        /// `η₁ = α − 1`
        eta1: f64,
        /// `η₂ = −β` — must be strictly negative.
        eta2: f64,
    },

    /// Dirichlet with natural parameters `η_k = α_k − 1`.
    ///
    /// The concentration parameters are `α_k = η_k + 1 > 0`.
    ///
    /// Sufficient statistics `T(x) = (ln x_1, …, ln x_K)` and the
    /// log-partition `A(η) = Σ ln Γ(η_k+1) − ln Γ(Σ(η_k+1))` give the
    /// standard exponential family structure.
    ///
    /// For binary variables (`K = 2`), this reduces to `Beta(α, β)`
    /// stored as `Dirichlet([α−1, β−1])`.
    ///
    /// Dimension `d = K`.
    Dirichlet {
        /// Natural parameters `η_k = α_k − 1`, length `K`.
        eta: Vec<f64>,
    },
}

impl NaturalParams {
    /// Dimension of the sufficient statistic vector `T(x)`.
    #[must_use]
    pub const fn suff_stat_dim(&self) -> usize {
        match self {
            Self::Gaussian { .. } | Self::Gamma { .. } => 2,
            Self::Dirichlet { eta } => eta.len(),
        }
    }

    /// Expected sufficient statistics `E_η[T(x)] = ∇A(η)`.
    ///
    /// This is the fundamental quantity for the coupling terms in the ELBO.
    #[must_use]
    pub fn expected_suff_stats(&self) -> DVector<f64> {
        match self {
            Self::Gaussian { eta1, eta2 } => gaussian::expected_suff_stats(*eta1, *eta2),
            Self::Gamma { eta1, eta2 } => gamma::expected_suff_stats(*eta1, *eta2),
            Self::Dirichlet { eta } => {
                let alpha: Vec<f64> = eta.iter().map(|&e| e + 1.0).collect();
                dirichlet::expected_suff_stats(&alpha)
            }
        }
    }

    /// Differential entropy `H(p_η) = −E_η[ln p_η(x)]`.
    ///
    /// Computed via the generic exponential family formula:
    /// `H = −E[ln h(x)] − η · E[T(x)] + A(η)`.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let eta = self.eta_vector();
        match self {
            Self::Gaussian { .. } => ef_entropy::<Gaussian>(&eta),
            Self::Gamma { .. } => ef_entropy::<GammaDist>(&eta),
            Self::Dirichlet { .. } => ef_entropy::<DirichletDist>(&eta),
        }
    }

    /// Fisher information matrix `F(η) = ∇²A(η) = Cov_η[T(x)]`.
    ///
    /// Returns a `d × d` matrix where `d = suff_stat_dim()`.
    #[must_use]
    pub fn fisher_information(&self) -> DMatrix<f64> {
        let eta = self.eta_vector();
        match self {
            Self::Gaussian { .. } => Gaussian::fisher_information(&eta),
            Self::Gamma { .. } => GammaDist::fisher_information(&eta),
            Self::Dirichlet { .. } => DirichletDist::fisher_information(&eta),
        }
    }

    /// Cross-entropy `E_self[ln p_other(x)]` where both distributions
    /// belong to the **same** family.
    ///
    /// Computed via the generic exponential family formula:
    /// `E_me[ln p_other] = E_me[ln h(x)] + η_other · E_me[T(x)] − A(η_other)`.
    ///
    /// Returns `NaN` if `self` and `other` are different families.
    #[must_use]
    pub fn cross_entropy(&self, other: &Self) -> f64 {
        let me = self.eta_vector();
        let oth = other.eta_vector();
        match (self, other) {
            (Self::Gaussian { .. }, Self::Gaussian { .. }) => {
                ef_cross_entropy::<Gaussian>(&me, &oth)
            }
            (Self::Gamma { .. }, Self::Gamma { .. }) => ef_cross_entropy::<GammaDist>(&me, &oth),
            (Self::Dirichlet { .. }, Self::Dirichlet { .. }) => {
                ef_cross_entropy::<DirichletDist>(&me, &oth)
            }
            // Mismatched families — return NaN (debug builds will catch this).
            (Self::Gaussian { .. } | Self::Gamma { .. } | Self::Dirichlet { .. }, _) => {
                debug_assert!(false, "cross-entropy requires same family");
                f64::NAN
            }
        }
    }

    /// Log-partition function `A(η)`.
    #[must_use]
    pub fn log_partition(&self) -> f64 {
        match self {
            Self::Gaussian { eta1, eta2 } => gaussian::log_partition(*eta1, *eta2),
            Self::Gamma { eta1, eta2 } => gamma::log_partition(*eta1, *eta2),
            Self::Dirichlet { eta } => {
                let alpha: Vec<f64> = eta.iter().map(|&e| e + 1.0).collect();
                dirichlet::log_partition(&alpha)
            }
        }
    }

    /// Natural parameter vector `η` as a dense vector.
    #[must_use]
    pub fn eta_vector(&self) -> DVector<f64> {
        match self {
            Self::Gaussian { eta1, eta2 } | Self::Gamma { eta1, eta2 } => {
                DVector::from_vec(vec![*eta1, *eta2])
            }
            Self::Dirichlet { eta } => DVector::from_vec(eta.clone()),
        }
    }

    /// Linear interpolation in natural parameter space.
    ///
    /// Returns `(1 − t) · self + t · other`.
    ///
    /// Returns a clone of `self` if `self` and `other` are different
    /// families or have different dimensions.
    #[must_use]
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let s = 1.0 - t;
        match (self, other) {
            (Self::Gaussian { eta1: a1, eta2: a2 }, Self::Gaussian { eta1: b1, eta2: b2 }) => {
                Self::Gaussian {
                    eta1: s * a1 + t * b1,
                    eta2: s * a2 + t * b2,
                }
            }
            (Self::Gamma { eta1: a1, eta2: a2 }, Self::Gamma { eta1: b1, eta2: b2 }) => {
                Self::Gamma {
                    eta1: s * a1 + t * b1,
                    eta2: s * a2 + t * b2,
                }
            }
            (Self::Dirichlet { eta: a }, Self::Dirichlet { eta: b }) => {
                debug_assert!(a.len() == b.len(), "dirichlet dimension mismatch");
                Self::Dirichlet {
                    eta: a
                        .iter()
                        .zip(b.iter())
                        .map(|(ai, bi)| s.mul_add(*ai, t * bi))
                        .collect(),
                }
            }
            // Mismatched families — return self unchanged (debug builds catch this).
            (Self::Gaussian { .. } | Self::Gamma { .. } | Self::Dirichlet { .. }, _) => {
                debug_assert!(false, "interpolation requires same family");
                self.clone()
            }
        }
    }

    /// Build `NaturalParams` from a natural parameter vector and a
    /// reference that determines the family and dimensions.
    ///
    /// The vector is projected onto the valid domain before construction.
    ///
    /// Returns `None` if `eta` has the wrong length for the family.
    #[must_use]
    pub fn from_eta_vector(eta: &DVector<f64>, reference: &Self) -> Option<Self> {
        let mut eta = eta.clone();
        // Project onto the valid domain for this family.
        match reference {
            Self::Gaussian { .. } => Gaussian::project(&mut eta),
            Self::Gamma { .. } => GammaDist::project(&mut eta),
            Self::Dirichlet { .. } => DirichletDist::project(&mut eta),
        }
        let s = eta.as_slice();
        match reference {
            Self::Gaussian { .. } => {
                let (&e1, &e2) = s.first().zip(s.get(1))?;
                (s.len() == 2).then_some(Self::Gaussian { eta1: e1, eta2: e2 })
            }
            Self::Gamma { .. } => {
                let (&e1, &e2) = s.first().zip(s.get(1))?;
                (s.len() == 2).then_some(Self::Gamma { eta1: e1, eta2: e2 })
            }
            Self::Dirichlet { eta: ref_eta } if s.len() == ref_eta.len() => {
                Some(Self::Dirichlet { eta: s.to_vec() })
            }
            Self::Dirichlet { .. } => None,
        }
    }

    /// Convert natural parameters back to canonical (human-readable) form.
    ///
    /// Returns a [`FamilyDef`] suitable for YAML serialization.
    #[must_use]
    pub fn to_canonical(&self) -> FamilyDef {
        match self {
            Self::Gaussian { eta1, eta2 } => {
                let sigma2 = -1.0 / (2.0 * eta2);
                FamilyDef::Gaussian {
                    mu: eta1 * sigma2,
                    sigma2,
                }
            }
            Self::Gamma { eta1, eta2 } => FamilyDef::Gamma {
                alpha: eta1 + 1.0,
                beta: -eta2,
            },
            Self::Dirichlet { eta } => {
                // K=2 → output as Beta sugar for readability.
                if eta.len() == 2
                    && let (Some(&e1), Some(&e2)) = (eta.first(), eta.get(1))
                {
                    FamilyDef::Beta {
                        alpha: e1 + 1.0,
                        beta: e2 + 1.0,
                    }
                } else {
                    FamilyDef::Dirichlet {
                        alpha: eta.iter().map(|&e| e + 1.0).collect(),
                    }
                }
            }
        }
    }
}
