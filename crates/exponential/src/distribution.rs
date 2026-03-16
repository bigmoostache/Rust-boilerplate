//! The [`ExponentialFamily`] trait: generic interface for any exponential family member.
//!
//! An exponential family distribution in canonical form is:
//!
//! ```text
//! p(x | η) = h(x) · exp(ηᵀ T(x) − A(η))
//! ```
//!
//! Implementors define the four components `(h, T, A, ∇A)` plus the dimension `k`
//! of the natural parameter space.

use nalgebra::DVector;

use crate::errors::ExponentialError;

/// A member of the exponential family in canonical form.
///
/// The observation type `X` is generic — it can be `f64` for univariate families,
/// `DVector<f64>` for multivariate ones, or any domain-specific type.
///
/// # Required methods
///
/// | Method | Math | Description |
/// |--------|------|-------------|
/// | [`dimension`] | `k` | Dimension of the natural parameter space |
/// | [`sufficient_statistic`] | `T(x)` | Maps an observation to ℝᵏ |
/// | [`log_partition`] | `A(η)` | Log-normalizer, ensures ∫ p(x\|η) dx = 1 |
/// | [`log_base_measure`] | `ln h(x)` | Log of the base measure (for numerical stability) |
///
/// # Provided methods
///
/// | Method | Math | Description |
/// |--------|------|-------------|
/// | [`log_partition_gradient`] | `∇A(η) = E[T(x)]` | Gradient of A, defaults to finite differences |
/// | [`log_density`] | `ln p(x\|η)` | Full log-density computation |
/// | [`validate_natural_parameter`] | — | Dimension check for η |
/// | [`validate_sufficient_statistic`] | — | Dimension check for T(x) |
///
/// [`dimension`]: ExponentialFamily::dimension
/// [`sufficient_statistic`]: ExponentialFamily::sufficient_statistic
/// [`log_partition`]: ExponentialFamily::log_partition
/// [`log_base_measure`]: ExponentialFamily::log_base_measure
/// [`log_partition_gradient`]: ExponentialFamily::log_partition_gradient
/// [`log_density`]: ExponentialFamily::log_density
/// [`validate_natural_parameter`]: ExponentialFamily::validate_natural_parameter
/// [`validate_sufficient_statistic`]: ExponentialFamily::validate_sufficient_statistic
pub trait ExponentialFamily<X> {
    /// Dimension `k` of the natural parameter space ℝᵏ.
    fn dimension(&self) -> usize;

    /// Sufficient statistic `T(x) → ℝᵏ`.
    ///
    /// Maps an observation `x` into the natural parameter space.
    ///
    /// # Errors
    ///
    /// Returns an error if the observation cannot be mapped (e.g. domain violation).
    fn sufficient_statistic(&self, x: &X) -> Result<DVector<f64>, ExponentialError>;

    /// Log-partition function `A(η)`.
    ///
    /// Ensures the distribution integrates to 1:
    /// `A(η) = ln ∫ h(x) exp(ηᵀ T(x)) dx`.
    ///
    /// # Errors
    ///
    /// Returns an error if `η` has wrong dimension or `A(η)` is not finite.
    fn log_partition(&self, eta: &DVector<f64>) -> Result<f64, ExponentialError>;

    /// Log of the base measure `ln h(x)`.
    ///
    /// We work in log-space for numerical stability. The actual base measure
    /// is `h(x) = exp(ln_h(x))`.
    ///
    /// # Errors
    ///
    /// Returns an error if the base measure is invalid for this observation.
    fn log_base_measure(&self, x: &X) -> Result<f64, ExponentialError>;

    /// Gradient of the log-partition: `∇A(η) = E_η[T(x)]`.
    ///
    /// The gradient of `A` at `η` gives the expected sufficient statistic
    /// under the distribution parameterized by `η`.
    ///
    /// The default implementation uses central finite differences with step `ε = 1e-7`.
    /// Override this for closed-form gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `η` has wrong dimension or any `A` evaluation fails.
    fn log_partition_gradient(&self, eta: &DVector<f64>) -> Result<DVector<f64>, ExponentialError> {
        self.validate_natural_parameter(eta)?;
        let k = self.dimension();
        let eps = 1e-7_f64;
        let mut grad = DVector::zeros(k);
        for i in 0..k {
            let mut eta_plus = eta.clone();
            let mut eta_minus = eta.clone();
            let p =
                eta_plus
                    .get_mut(i)
                    .ok_or(ExponentialError::NaturalParameterDimensionMismatch {
                        expected: k,
                        got: i,
                    })?;
            *p += eps;
            let m = eta_minus.get_mut(i).ok_or(
                ExponentialError::NaturalParameterDimensionMismatch {
                    expected: k,
                    got: i,
                },
            )?;
            *m -= eps;
            let a_plus = self.log_partition(&eta_plus)?;
            let a_minus = self.log_partition(&eta_minus)?;
            let g = grad
                .get_mut(i)
                .ok_or(ExponentialError::NaturalParameterDimensionMismatch {
                    expected: k,
                    got: i,
                })?;
            *g = (a_plus - a_minus) / (2.0 * eps);
        }
        Ok(grad)
    }

    /// Log-density `ln p(x | η) = ln h(x) + ηᵀ T(x) − A(η)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any component computation fails.
    fn log_density(&self, x: &X, eta: &DVector<f64>) -> Result<f64, ExponentialError> {
        self.validate_natural_parameter(eta)?;
        let t = self.sufficient_statistic(x)?;
        let a = self.log_partition(eta)?;
        let ln_h = self.log_base_measure(x)?;
        Ok(ln_h + eta.dot(&t) - a)
    }

    /// Validate that `η` has the correct dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ExponentialError::NaturalParameterDimensionMismatch`] if
    /// `eta.len() != self.dimension()`.
    fn validate_natural_parameter(&self, eta: &DVector<f64>) -> Result<(), ExponentialError> {
        let k = self.dimension();
        if eta.len() == k {
            Ok(())
        } else {
            Err(ExponentialError::NaturalParameterDimensionMismatch {
                expected: k,
                got: eta.len(),
            })
        }
    }

    /// Validate that a sufficient statistic vector has the correct dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ExponentialError::SufficientStatisticDimensionMismatch`] if
    /// `t.len() != self.dimension()`.
    fn validate_sufficient_statistic(&self, t: &DVector<f64>) -> Result<(), ExponentialError> {
        let k = self.dimension();
        if t.len() == k {
            Ok(())
        } else {
            Err(ExponentialError::SufficientStatisticDimensionMismatch {
                expected: k,
                got: t.len(),
            })
        }
    }
}
