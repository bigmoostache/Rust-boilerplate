//! 1-D Normal (Gaussian) distribution as an exponential family member.
//!
//! # Standard parameterization
//!
//! `N(μ, σ²)` with mean `μ` and variance `σ² > 0`.
//!
//! # Exponential family form
//!
//! ```text
//! p(x | η) = h(x) · exp(ηᵀ T(x) − A(η))
//! ```
//!
//! | Component | Formula |
//! |-----------|---------|
//! | `η` | `(μ / σ², −1 / (2σ²))` |
//! | `T(x)` | `(x, x²)` |
//! | `A(η)` | `−η₁² / (4 η₂) + ½ ln(−π / η₂)` |
//! | `h(x)` | `1` (so `ln h(x) = 0`) |
//!
//! The natural parameter space requires `η₂ < 0` (ensures finite variance).
//!
//! # Conversions
//!
//! | Direction | Formula |
//! |-----------|---------|
//! | `(μ, σ²) → η` | `η₁ = μ / σ²`, `η₂ = −1 / (2 σ²)` |
//! | `η → (μ, σ²)` | `σ² = −1 / (2 η₂)`, `μ = −η₁ / (2 η₂)` |

use core::f64::consts::PI;

use nalgebra::DVector;

use crate::distribution::ExponentialFamily;
use crate::errors::ExponentialError;

/// Dimension of the 1-D Normal natural parameter space.
const DIM: usize = 2;

/// Extract the two natural parameters `(η₁, η₂)` from `η ∈ ℝ²`.
///
/// # Errors
///
/// Returns [`ExponentialError::NaturalParameterDimensionMismatch`] if `η.len() ≠ 2`.
fn unpack_eta(eta: &DVector<f64>) -> Result<(f64, f64), ExponentialError> {
    if eta.len() != DIM {
        return Err(ExponentialError::NaturalParameterDimensionMismatch {
            expected: DIM,
            got: eta.len(),
        });
    }
    let eta1 = *eta
        .get(0)
        .ok_or(ExponentialError::NaturalParameterDimensionMismatch {
            expected: DIM,
            got: 0,
        })?;
    let eta2 = *eta
        .get(1)
        .ok_or(ExponentialError::NaturalParameterDimensionMismatch {
            expected: DIM,
            got: 1,
        })?;
    Ok((eta1, eta2))
}

/// 1-D Normal (Gaussian) distribution in exponential family form.
///
/// Stores the standard parameters `(μ, σ²)` and implements the
/// [`ExponentialFamily<f64>`] trait with `k = 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal1D {
    /// Mean `μ`.
    mu: f64,
    /// Variance `σ² > 0`.
    sigma_sq: f64,
}

impl Normal1D {
    /// Create a new 1-D Normal distribution with mean `μ` and variance `σ²`.
    ///
    /// # Errors
    ///
    /// Returns [`ExponentialError::NonPositivePseudoCount`] if `σ² ≤ 0`.
    pub fn new(mu: f64, sigma_sq: f64) -> Result<Self, ExponentialError> {
        if sigma_sq <= 0.0 || !sigma_sq.is_finite() {
            return Err(ExponentialError::NonPositivePseudoCount(sigma_sq));
        }
        if !mu.is_finite() {
            return Err(ExponentialError::NonFiniteLogPartition(mu));
        }
        Ok(Self { mu, sigma_sq })
    }

    /// Mean `μ`.
    #[must_use]
    pub const fn mu(&self) -> f64 {
        self.mu
    }

    /// Variance `σ²`.
    #[must_use]
    pub const fn sigma_sq(&self) -> f64 {
        self.sigma_sq
    }

    /// Standard deviation `σ`.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.sigma_sq.sqrt()
    }

    /// Convert standard parameters `(μ, σ²)` to natural parameters `η ∈ ℝ²`.
    ///
    /// ```text
    /// η₁ = μ / σ²
    /// η₂ = −1 / (2 σ²)
    /// ```
    #[must_use]
    pub fn to_natural(&self) -> DVector<f64> {
        let eta1 = self.mu / self.sigma_sq;
        let eta2 = -1.0 / (2.0 * self.sigma_sq);
        DVector::from_vec(vec![eta1, eta2])
    }

    /// Recover standard parameters `(μ, σ²)` from natural parameters `η`.
    ///
    /// ```text
    /// σ² = −1 / (2 η₂)
    /// μ  = −η₁ / (2 η₂)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `η` does not have dimension 2
    /// - `η₂ ≥ 0` (would give non-positive variance)
    pub fn from_natural(eta: &DVector<f64>) -> Result<Self, ExponentialError> {
        let (eta1, eta2) = unpack_eta(eta)?;
        if eta2 >= 0.0 || !eta2.is_finite() {
            return Err(ExponentialError::NonPositivePseudoCount(eta2));
        }
        let sigma_sq = -1.0 / (2.0 * eta2);
        let mu = eta1 * sigma_sq;
        Self::new(mu, sigma_sq)
    }
}

impl ExponentialFamily<f64> for Normal1D {
    fn dimension(&self) -> usize {
        DIM
    }

    fn sufficient_statistic(&self, x: &f64) -> Result<DVector<f64>, ExponentialError> {
        if !x.is_finite() {
            return Err(ExponentialError::InvalidBaseMeasure(*x));
        }
        Ok(DVector::from_vec(vec![*x, x * x]))
    }

    fn log_partition(&self, eta: &DVector<f64>) -> Result<f64, ExponentialError> {
        self.validate_natural_parameter(eta)?;
        let (eta1, eta2) = unpack_eta(eta)?;
        if eta2 >= 0.0 {
            return Err(ExponentialError::NonFiniteLogPartition(eta2));
        }
        // A(η) = −η₁² / (4 η₂) + ½ ln(−π / η₂)
        let term1 = -(eta1 * eta1) / (4.0 * eta2);
        let term2 = 0.5 * (-PI / eta2).ln();
        let result = term1 + term2;
        if !result.is_finite() {
            return Err(ExponentialError::NonFiniteLogPartition(result));
        }
        Ok(result)
    }

    fn log_base_measure(&self, _x: &f64) -> Result<f64, ExponentialError> {
        Ok(0.0)
    }

    fn log_partition_gradient(&self, eta: &DVector<f64>) -> Result<DVector<f64>, ExponentialError> {
        self.validate_natural_parameter(eta)?;
        let (eta1, eta2) = unpack_eta(eta)?;
        if eta2 >= 0.0 {
            return Err(ExponentialError::NonFiniteLogPartition(eta2));
        }
        // ∂A/∂η₁ = −η₁ / (2 η₂)  =  μ  =  E[x]
        let grad1 = -eta1 / (2.0 * eta2);
        // ∂A/∂η₂ = η₁² / (4 η₂²) − 1/(2 η₂)  =  μ² + σ²  =  E[x²]
        let grad2 = (eta1 * eta1) / (4.0 * eta2 * eta2) - 1.0 / (2.0 * eta2);
        Ok(DVector::from_vec(vec![grad1, grad2]))
    }

    fn log_density(&self, x: &f64, eta: &DVector<f64>) -> Result<f64, ExponentialError> {
        self.validate_natural_parameter(eta)?;
        let t = self.sufficient_statistic(x)?;
        let a = self.log_partition(eta)?;
        let ln_h = self.log_base_measure(x)?;
        Ok(ln_h + eta.dot(&t) - a)
    }

    fn validate_natural_parameter(&self, eta: &DVector<f64>) -> Result<(), ExponentialError> {
        if eta.len() == DIM {
            Ok(())
        } else {
            Err(ExponentialError::NaturalParameterDimensionMismatch {
                expected: DIM,
                got: eta.len(),
            })
        }
    }

    fn validate_sufficient_statistic(&self, t: &DVector<f64>) -> Result<(), ExponentialError> {
        if t.len() == DIM {
            Ok(())
        } else {
            Err(ExponentialError::SufficientStatisticDimensionMismatch {
                expected: DIM,
                got: t.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOLERANCE
    }

    #[test]
    fn roundtrip_standard_to_natural_and_back() {
        let original = Normal1D::new(3.0, 4.0).unwrap();
        let eta = original.to_natural();
        let recovered = Normal1D::from_natural(&eta).unwrap();
        assert!(approx_eq(original.mu(), recovered.mu()));
        assert!(approx_eq(original.sigma_sq(), recovered.sigma_sq()));
    }

    #[test]
    fn natural_parameter_values() {
        // μ=2, σ²=0.5 → η₁ = 2/0.5 = 4, η₂ = -1/(2·0.5) = -1
        let n = Normal1D::new(2.0, 0.5).unwrap();
        let eta = n.to_natural();
        let (eta1, eta2) = unpack_eta(&eta).unwrap();
        assert!(approx_eq(eta1, 4.0));
        assert!(approx_eq(eta2, -1.0));
    }

    #[test]
    fn sufficient_statistic_values() {
        let n = Normal1D::new(0.0, 1.0).unwrap();
        let t = n.sufficient_statistic(&3.0).unwrap();
        let t0 = *t.get(0).unwrap();
        let t1 = *t.get(1).unwrap();
        assert!(approx_eq(t0, 3.0));
        assert!(approx_eq(t1, 9.0));
    }

    #[test]
    fn log_density_matches_direct_computation() {
        let n = Normal1D::new(2.0, 3.0).unwrap();
        let x = 1.5_f64;
        let eta = n.to_natural();
        let log_p = n.log_density(&x, &eta).unwrap();

        // Direct: ln N(x|μ,σ²) = -½ ln(2πσ²) - (x-μ)²/(2σ²)
        let direct = -0.5 * (2.0 * PI * 3.0).ln() - (1.5 - 2.0) * (1.5 - 2.0) / (2.0 * 3.0);
        assert!(
            approx_eq(log_p, direct),
            "log_density={log_p}, direct={direct}"
        );
    }

    #[test]
    fn gradient_matches_expected_sufficient_statistics() {
        // For N(μ, σ²): ∇A(η) = (E[x], E[x²]) = (μ, μ² + σ²)
        let n = Normal1D::new(5.0, 2.0).unwrap();
        let eta = n.to_natural();
        let grad = n.log_partition_gradient(&eta).unwrap();
        let g0 = *grad.get(0).unwrap();
        let g1 = *grad.get(1).unwrap();
        assert!(approx_eq(g0, 5.0)); // E[x] = μ
        assert!(approx_eq(g1, 27.0)); // E[x²] = μ² + σ² = 25 + 2
    }

    #[test]
    fn dimension_is_two() {
        let n = Normal1D::new(0.0, 1.0).unwrap();
        assert_eq!(n.dimension(), 2);
    }

    #[test]
    fn base_measure_is_one() {
        let n = Normal1D::new(0.0, 1.0).unwrap();
        assert!(approx_eq(n.log_base_measure(&42.0).unwrap(), 0.0));
    }

    #[test]
    fn invalid_variance_rejected() {
        assert!(Normal1D::new(0.0, 0.0).is_err());
        assert!(Normal1D::new(0.0, -1.0).is_err());
        assert!(Normal1D::new(0.0, f64::INFINITY).is_err());
        assert!(Normal1D::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn from_natural_rejects_non_negative_eta2() {
        let bad_eta = DVector::from_vec(vec![1.0, 0.0]);
        assert!(Normal1D::from_natural(&bad_eta).is_err());
        let bad_eta2 = DVector::from_vec(vec![1.0, 1.0]);
        assert!(Normal1D::from_natural(&bad_eta2).is_err());
    }

    #[test]
    fn from_natural_rejects_wrong_dimension() {
        let bad = DVector::from_vec(vec![1.0]);
        assert!(Normal1D::from_natural(&bad).is_err());
        let bad3 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(Normal1D::from_natural(&bad3).is_err());
    }
}
