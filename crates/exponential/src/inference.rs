//! The [`ConjugatePrior`] struct: universal conjugate prior for exponential families.
//!
//! For any exponential family, the conjugate prior has the form:
//!
//! ```text
//! p(η | λ, ν) ∝ exp(ηᵀ λ − ν A(η))
//! ```
//!
//! where `(λ ∈ ℝᵏ, ν > 0)` are hyperparameters with interpretation:
//! - `λ`: pseudo-sufficient statistic (prior belief about Σ T(xᵢ))
//! - `ν`: pseudo-observation count
//! - `λ / ν`: prior expectation of `E[T(x)]`
//!
//! Bayesian updating is additive:
//! - `λₙ = λ₀ + Σᵢ T(xᵢ)`
//! - `νₙ = ν₀ + n`

use nalgebra::DVector;

use crate::distribution::ExponentialFamily;
use crate::errors::ExponentialError;

/// Universal conjugate prior for an exponential family.
///
/// Holds the hyperparameters `(λ, ν)` and provides Bayesian updating,
/// log-density evaluation, and posterior mean computation.
///
/// # Type parameter
///
/// - `X`: the observation type of the underlying [`ExponentialFamily`].
#[derive(Debug, Clone)]
pub struct ConjugatePrior {
    /// Pseudo-sufficient statistic `λ ∈ ℝᵏ`.
    lambda: DVector<f64>,
    /// Pseudo-observation count `ν > 0`.
    nu: f64,
}

impl ConjugatePrior {
    /// Create a new conjugate prior with hyperparameters `(λ, ν)`.
    ///
    /// # Errors
    ///
    /// Returns [`ExponentialError::NonPositivePseudoCount`] if `ν ≤ 0`.
    pub fn new(lambda: DVector<f64>, nu: f64) -> Result<Self, ExponentialError> {
        if nu <= 0.0 {
            return Err(ExponentialError::NonPositivePseudoCount(nu));
        }
        Ok(Self { lambda, nu })
    }

    /// The pseudo-sufficient statistic `λ`.
    #[must_use]
    pub const fn lambda(&self) -> &DVector<f64> {
        &self.lambda
    }

    /// The pseudo-observation count `ν`.
    #[must_use]
    pub const fn nu(&self) -> f64 {
        self.nu
    }

    /// Prior mean of the sufficient statistic: `λ / ν`.
    ///
    /// This is the prior expectation `E[T(x)]` implied by the hyperparameters.
    #[must_use]
    pub fn prior_mean(&self) -> DVector<f64> {
        self.lambda.map(|v| v / self.nu)
    }

    /// Unnormalized log-density of the prior at natural parameter `η`:
    ///
    /// ```text
    /// ln p(η | λ, ν) = ηᵀ λ − ν A(η) + const
    /// ```
    ///
    /// The additive constant (log-normalizer of the prior itself) is omitted
    /// since it doesn't depend on `η`.
    ///
    /// # Errors
    ///
    /// Returns an error if `η` has wrong dimension or `A(η)` evaluation fails.
    pub fn unnormalized_log_density<X, F: ExponentialFamily<X>>(
        &self,
        family: &F,
        eta: &DVector<f64>,
    ) -> Result<f64, ExponentialError> {
        family.validate_natural_parameter(eta)?;
        if eta.len() != self.lambda.len() {
            return Err(ExponentialError::NaturalParameterDimensionMismatch {
                expected: self.lambda.len(),
                got: eta.len(),
            });
        }
        let a = family.log_partition(eta)?;
        Ok(self.nu.mul_add(-a, eta.dot(&self.lambda)))
    }

    /// Update the prior with a single observation, returning the posterior.
    ///
    /// ```text
    /// λₙ = λ + T(x)
    /// νₙ = ν + 1
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the sufficient statistic computation fails
    /// or its dimension doesn't match.
    pub fn update_single<X, F: ExponentialFamily<X>>(
        &self,
        family: &F,
        x: &X,
    ) -> Result<Self, ExponentialError> {
        let t = family.sufficient_statistic(x)?;
        family.validate_sufficient_statistic(&t)?;
        if t.len() != self.lambda.len() {
            return Err(ExponentialError::SufficientStatisticDimensionMismatch {
                expected: self.lambda.len(),
                got: t.len(),
            });
        }
        let new_lambda = self.lambda.zip_map(&t, |a: f64, b: f64| a + b);
        Ok(Self {
            lambda: new_lambda,
            nu: self.nu + 1.0,
        })
    }

    /// Update the prior with a batch of observations, returning the posterior.
    ///
    /// ```text
    /// λₙ = λ₀ + Σᵢ T(xᵢ)
    /// νₙ = ν₀ + n
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if any sufficient statistic computation fails
    /// or dimensions don't match.
    pub fn update_batch<X, F: ExponentialFamily<X>>(
        &self,
        family: &F,
        observations: &[X],
    ) -> Result<Self, ExponentialError> {
        let mut lambda_sum: DVector<f64> = DVector::zeros(self.lambda.len());
        for x in observations {
            let t = family.sufficient_statistic(x)?;
            family.validate_sufficient_statistic(&t)?;
            if t.len() != self.lambda.len() {
                return Err(ExponentialError::SufficientStatisticDimensionMismatch {
                    expected: self.lambda.len(),
                    got: t.len(),
                });
            }
            lambda_sum = lambda_sum.zip_map(&t, |a: f64, b: f64| a + b);
        }
        let n = observations.len();
        let n_f64 = f64::from(
            u32::try_from(n).map_err(|_err| ExponentialError::ObservationCountOverflow(n))?,
        );
        let new_lambda = self.lambda.zip_map(&lambda_sum, |a: f64, b: f64| a + b);
        Ok(Self {
            lambda: new_lambda,
            nu: self.nu + n_f64,
        })
    }

    /// Posterior mean of the sufficient statistic: `λₙ / νₙ`.
    ///
    /// Equivalent to calling [`prior_mean`](Self::prior_mean) on the posterior,
    /// which gives the convex combination:
    ///
    /// ```text
    /// λₙ / νₙ = (ν₀ / νₙ) · (λ₀ / ν₀) + (n / νₙ) · T̄
    /// ```
    ///
    /// where `T̄ = (1/n) Σ T(xᵢ)` is the sample mean of sufficient statistics.
    #[must_use]
    pub fn posterior_mean(&self) -> DVector<f64> {
        self.prior_mean()
    }

    /// Compute the shrinkage weight toward the prior: `ν₀ / (ν₀ + n)`.
    ///
    /// Returns a value in `(0, 1]` indicating how much the posterior
    /// is influenced by the prior vs. the data.
    ///
    /// # Arguments
    ///
    /// - `n`: number of observations incorporated.
    ///
    /// # Errors
    ///
    /// Returns [`ExponentialError::ObservationCountOverflow`] if `n > u32::MAX`.
    pub fn shrinkage_weight(&self, n: usize) -> Result<f64, ExponentialError> {
        let n_f = f64::from(
            u32::try_from(n).map_err(|_err| ExponentialError::ObservationCountOverflow(n))?,
        );
        Ok(self.nu / (self.nu + n_f))
    }
}
