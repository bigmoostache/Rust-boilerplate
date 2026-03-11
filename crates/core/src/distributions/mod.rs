//! Exponential family distributions in natural parameterization.
//!
//! Each variant of [`NaturalParams`] represents a distribution from an
//! exponential family, stored in its natural (canonical) parameterization.
//! All key quantities — expected sufficient statistics, entropy,
//! cross-entropy, log-partition — are computed in closed form.

mod bernoulli;
mod beta;
mod categorical;
mod dirichlet;
pub(crate) mod gamma;
mod gaussian;
mod poisson;

use nalgebra::DVector;

use crate::schema::raw::FamilyDef;

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

    /// Beta with natural parameters `(η₁, η₂)`.
    ///
    /// - `η₁ = α − 1`  (must be > −1)
    /// - `η₂ = β − 1`  (must be > −1)
    /// - Sufficient statistics: `T(x) = (ln x, ln(1−x))`, dimension `d = 2`
    Beta {
        /// `η₁ = α − 1`
        eta1: f64,
        /// `η₂ = β − 1`
        eta2: f64,
    },

    /// Poisson with natural parameter `η₁ = ln λ`.
    ///
    /// - Sufficient statistic: `T(x) = x`, dimension `d = 1`
    Poisson {
        /// `η₁ = ln λ`
        eta1: f64,
    },

    /// Bernoulli with natural parameter `η₁ = logit(p)`.
    ///
    /// - Sufficient statistic: `T(x) = x`, dimension `d = 1`
    Bernoulli {
        /// `η₁ = logit(p) = ln(p / (1−p))`
        eta1: f64,
    },

    /// Categorical over `K` classes, stored as `K − 1` log-ratios.
    ///
    /// - `η_k = ln(p_k / p_K)` for `k = 1, …, K−1`
    /// - Sufficient statistic: `T(x) = (𝟙{x=1}, …, 𝟙{x=K−1})`, dim `d = K−1`
    Categorical {
        /// Log-ratios `η_k = ln(p_k / p_K)`, length `K − 1`.
        eta: Vec<f64>,
    },

    /// Dirichlet with concentration parameters `α_k > 0`.
    ///
    /// Not a standard exponential family in the textbook sense, but the
    /// sufficient statistics `T(x) = (ln x_1, …, ln x_K)` and the
    /// log-partition `A(α) = Σ ln Γ(α_k) − ln Γ(Σ α_k)` give the same
    /// algebraic structure we need.
    ///
    /// Dimension `d = K`.
    Dirichlet {
        /// Concentration parameters `α_k > 0`, length `K`.
        alpha: Vec<f64>,
    },
}

impl NaturalParams {
    /// Dimension of the sufficient statistic vector `T(x)`.
    #[must_use]
    pub const fn suff_stat_dim(&self) -> usize {
        match self {
            Self::Gaussian { .. } | Self::Gamma { .. } | Self::Beta { .. } => 2,
            Self::Poisson { .. } | Self::Bernoulli { .. } => 1,
            Self::Categorical { eta } => eta.len(),
            Self::Dirichlet { alpha } => alpha.len(),
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
            Self::Beta { eta1, eta2 } => beta::expected_suff_stats(*eta1, *eta2),
            Self::Poisson { eta1 } => poisson::expected_suff_stats(*eta1),
            Self::Bernoulli { eta1 } => bernoulli::expected_suff_stats(*eta1),
            Self::Categorical { eta } => categorical::expected_suff_stats(eta),
            Self::Dirichlet { alpha } => dirichlet::expected_suff_stats(alpha),
        }
    }

    /// Differential entropy `H(p_η) = −E_η[ln p_η(x)]`.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        match self {
            Self::Gaussian { eta1, eta2 } => gaussian::entropy(*eta1, *eta2),
            Self::Gamma { eta1, eta2 } => gamma::entropy(*eta1, *eta2),
            Self::Beta { eta1, eta2 } => beta::entropy(*eta1, *eta2),
            Self::Poisson { eta1 } => poisson::entropy(*eta1),
            Self::Bernoulli { eta1 } => bernoulli::entropy(*eta1),
            Self::Categorical { eta } => categorical::entropy(eta),
            Self::Dirichlet { alpha } => dirichlet::entropy(alpha),
        }
    }

    /// Cross-entropy `E_self[ln p_other(x)]` where both distributions
    /// belong to the **same** family.
    ///
    /// This is the prior term in the ELBO: how well the posterior
    /// explains data drawn from `other` (the relaxed prior).
    ///
    /// Returns `NaN` if `self` and `other` are different families.
    #[must_use]
    pub fn cross_entropy(&self, other: &Self) -> f64 {
        match (self, other) {
            (Self::Gaussian { eta1: a1, eta2: a2 }, Self::Gaussian { eta1: b1, eta2: b2 }) => {
                gaussian::cross_entropy(*a1, *a2, *b1, *b2)
            }
            (Self::Gamma { eta1: a1, eta2: a2 }, Self::Gamma { eta1: b1, eta2: b2 }) => {
                gamma::cross_entropy(*a1, *a2, *b1, *b2)
            }
            (Self::Beta { eta1: a1, eta2: a2 }, Self::Beta { eta1: b1, eta2: b2 }) => {
                beta::cross_entropy(*a1, *a2, *b1, *b2)
            }
            (Self::Poisson { eta1: a }, Self::Poisson { eta1: b }) => {
                poisson::cross_entropy(*a, *b)
            }
            (Self::Bernoulli { eta1: a }, Self::Bernoulli { eta1: b }) => {
                bernoulli::cross_entropy(*a, *b)
            }
            (Self::Categorical { eta: a }, Self::Categorical { eta: b }) => {
                categorical::cross_entropy(a, b)
            }
            (Self::Dirichlet { alpha: a }, Self::Dirichlet { alpha: b }) => {
                dirichlet::cross_entropy(a, b)
            }
            // Mismatched families — return NaN (debug builds will catch this).
            (
                Self::Gaussian { .. }
                | Self::Gamma { .. }
                | Self::Beta { .. }
                | Self::Poisson { .. }
                | Self::Bernoulli { .. }
                | Self::Categorical { .. }
                | Self::Dirichlet { .. },
                _,
            ) => {
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
            Self::Beta { eta1, eta2 } => beta::log_partition(*eta1, *eta2),
            Self::Poisson { eta1 } => poisson::log_partition(*eta1),
            Self::Bernoulli { eta1 } => bernoulli::log_partition(*eta1),
            Self::Categorical { eta } => categorical::log_partition(eta),
            Self::Dirichlet { alpha } => dirichlet::log_partition(alpha),
        }
    }

    /// Natural parameter vector `η` as a dense vector.
    #[must_use]
    pub fn eta_vector(&self) -> DVector<f64> {
        match self {
            Self::Gaussian { eta1, eta2 }
            | Self::Gamma { eta1, eta2 }
            | Self::Beta { eta1, eta2 } => DVector::from_vec(vec![*eta1, *eta2]),
            Self::Poisson { eta1 } | Self::Bernoulli { eta1 } => DVector::from_vec(vec![*eta1]),
            Self::Categorical { eta } => DVector::from_vec(eta.clone()),
            Self::Dirichlet { alpha } => DVector::from_vec(alpha.clone()),
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
            (Self::Beta { eta1: a1, eta2: a2 }, Self::Beta { eta1: b1, eta2: b2 }) => Self::Beta {
                eta1: s * a1 + t * b1,
                eta2: s * a2 + t * b2,
            },
            (Self::Poisson { eta1: a }, Self::Poisson { eta1: b }) => Self::Poisson {
                eta1: s * a + t * b,
            },
            (Self::Bernoulli { eta1: a }, Self::Bernoulli { eta1: b }) => Self::Bernoulli {
                eta1: s * a + t * b,
            },
            (Self::Categorical { eta: a }, Self::Categorical { eta: b }) => {
                debug_assert!(a.len() == b.len(), "categorical dimension mismatch");
                Self::Categorical {
                    eta: a
                        .iter()
                        .zip(b.iter())
                        .map(|(ai, bi)| s.mul_add(*ai, t * bi))
                        .collect(),
                }
            }
            (Self::Dirichlet { alpha: a }, Self::Dirichlet { alpha: b }) => {
                debug_assert!(a.len() == b.len(), "dirichlet dimension mismatch");
                Self::Dirichlet {
                    alpha: a
                        .iter()
                        .zip(b.iter())
                        .map(|(ai, bi)| s.mul_add(*ai, t * bi))
                        .collect(),
                }
            }
            // Mismatched families — return self unchanged (debug builds catch this).
            (
                Self::Gaussian { .. }
                | Self::Gamma { .. }
                | Self::Beta { .. }
                | Self::Poisson { .. }
                | Self::Bernoulli { .. }
                | Self::Categorical { .. }
                | Self::Dirichlet { .. },
                _,
            ) => {
                debug_assert!(false, "interpolation requires same family");
                self.clone()
            }
        }
    }

    /// Build `NaturalParams` from a natural parameter vector and a
    /// reference that determines the family and dimensions.
    ///
    /// Returns `None` if `eta` has the wrong length for the family.
    #[must_use]
    pub fn from_eta_vector(eta: &DVector<f64>, reference: &Self) -> Option<Self> {
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
            Self::Beta { .. } => {
                let (&e1, &e2) = s.first().zip(s.get(1))?;
                (s.len() == 2).then_some(Self::Beta { eta1: e1, eta2: e2 })
            }
            Self::Poisson { .. } => {
                let &e1 = s.first()?;
                (s.len() == 1).then_some(Self::Poisson { eta1: e1 })
            }
            Self::Bernoulli { .. } => {
                let &e1 = s.first()?;
                (s.len() == 1).then_some(Self::Bernoulli { eta1: e1 })
            }
            Self::Categorical { eta: ref_eta } if s.len() == ref_eta.len() => {
                Some(Self::Categorical { eta: s.to_vec() })
            }
            Self::Dirichlet { alpha: ref_alpha } if s.len() == ref_alpha.len() => {
                Some(Self::Dirichlet { alpha: s.to_vec() })
            }
            Self::Categorical { .. } | Self::Dirichlet { .. } => None,
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
            Self::Beta { eta1, eta2 } => FamilyDef::Beta {
                alpha: eta1 + 1.0,
                beta: eta2 + 1.0,
            },
            Self::Poisson { eta1 } => FamilyDef::Poisson { lambda: eta1.exp() },
            Self::Bernoulli { eta1 } => {
                let p = if *eta1 >= 0.0 {
                    1.0 / (1.0 + (-eta1).exp())
                } else {
                    let e = eta1.exp();
                    e / (1.0 + e)
                };
                FamilyDef::Bernoulli { p }
            }
            Self::Categorical { eta } => {
                // Recover probabilities from log-ratios via softmax
                let max_eta = eta.iter().copied().fold(0.0_f64, f64::max);
                let sum_exp: f64 =
                    eta.iter().map(|&e| (e - max_eta).exp()).sum::<f64>() + (-max_eta).exp();
                let log_z = max_eta + sum_exp.ln();
                let mut probs: Vec<f64> = eta.iter().map(|&e| (e - log_z).exp()).collect();
                probs.push((-log_z).exp());
                FamilyDef::Categorical { probs }
            }
            Self::Dirichlet { alpha } => FamilyDef::Dirichlet {
                alpha: alpha.clone(),
            },
        }
    }
}
