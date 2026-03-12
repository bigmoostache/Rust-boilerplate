//! Sufficient statistics and observation-point helpers.
//!
//! Computes `T(x)` for each exponential family at a concrete sample
//! point `x`.  Also provides conversions from `ObsPoint` / `TargetMean`
//! to typed values, and the canonical-to-natural mapping for `FamilyDef`.

use crate::schema::raw::FamilyDef;

use super::parse::{ObsPoint, TargetMean};

// ───────────────────────────────────────────────────────────────────
// T_B(x_B) — sufficient statistics at a point
// ───────────────────────────────────────────────────────────────────

/// Compute the sufficient statistic vector `T(x)` for a given family
/// at a sample-space point `x`.
///
/// Returns a `Vec<f64>` where `−∞` entries signal that the
/// corresponding column of `B` must be zero.
///
/// # Families
///
/// | Family      | `T(x)`                              |
/// |-------------|-------------------------------------|
/// | Gaussian    | `(x, x²)`                          |
/// | Gamma       | `(ln x, x)`                        |
/// | Dirichlet K | `(ln x₁, …, ln x_K)`               |
///
/// # Errors
///
/// Returns `Err` if `x` has wrong dimension for the family.
pub fn at_point(family: &FamilyDef, x: &ObsPoint) -> Result<Vec<f64>, String> {
    match family {
        FamilyDef::Gaussian { .. } => {
            let v = scalar(x, "Gaussian")?;
            Ok(vec![v, v * v])
        }
        FamilyDef::Gamma { .. } => {
            let v = scalar(x, "Gamma")?;
            if v < 0.0 {
                return Err("Gamma sample-space point must be ≥ 0".into());
            }
            Ok(vec![v.ln(), v])
        }
        FamilyDef::Beta { .. } => {
            let (p, q) = beta_point(x)?;
            Ok(vec![p.ln(), q.ln()])
        }
        FamilyDef::Dirichlet { alpha } => {
            let k = alpha.len();
            let pts = vec_point(x, k, "Dirichlet")?;
            Ok(pts.iter().map(|&v| v.ln()).collect())
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// ObsPoint extraction
// ───────────────────────────────────────────────────────────────────

/// Extract a scalar from an `ObsPoint`.
///
/// # Errors
///
/// Returns `Err` if the point is a vector with length ≠ 1.
pub fn scalar(x: &ObsPoint, family: &str) -> Result<f64, String> {
    match x {
        ObsPoint::Scalar(v) => Ok(*v),
        ObsPoint::Vec(v) if v.len() == 1 => v
            .first()
            .copied()
            .ok_or_else(|| format!("{family} x_b: empty vector")),
        ObsPoint::Vec(v) => Err(format!(
            "{family} x_b: expected scalar, got vector of length {}",
            v.len()
        )),
    }
}

/// Extract a Beta (Dirichlet K=2) simplex point from an `ObsPoint`.
///
/// # Errors
///
/// Returns `Err` if the vector has length ≠ 2.
pub fn beta_point(x: &ObsPoint) -> Result<(f64, f64), String> {
    match x {
        ObsPoint::Scalar(v) => Ok((*v, 1.0 - *v)),
        ObsPoint::Vec(v) if v.len() == 2 => {
            let p = v.first().copied().unwrap_or(0.0);
            let q = v.get(1).copied().unwrap_or(0.0);
            Ok((p, q))
        }
        ObsPoint::Vec(v) => Err(format!("Beta x_b: expected 2 values, got {}", v.len())),
    }
}

/// Extract a vector of length `k` from an `ObsPoint`.
///
/// # Errors
///
/// Returns `Err` if the vector length doesn't match `k`.
pub fn vec_point(x: &ObsPoint, k: usize, family: &str) -> Result<Vec<f64>, String> {
    match x {
        ObsPoint::Vec(v) if v.len() == k => Ok(v.clone()),
        ObsPoint::Vec(v) => Err(format!(
            "{family} x_b: expected {k} values, got {}",
            v.len()
        )),
        ObsPoint::Scalar(_) if k == 1 => {
            let v = scalar(x, family)?;
            Ok(vec![v])
        }
        ObsPoint::Scalar(_) => Err(format!("{family} x_b: expected {k} values, got scalar")),
    }
}

// ───────────────────────────────────────────────────────────────────
// Natural parameters from canonical
// ───────────────────────────────────────────────────────────────────

/// Compute `η_A` from canonical parameters of family A.
#[must_use]
pub fn eta_from_canonical(family: &FamilyDef) -> Vec<f64> {
    match family {
        FamilyDef::Gaussian { mu, sigma2 } => {
            vec![mu / sigma2, -1.0 / (2.0 * sigma2)]
        }
        FamilyDef::Gamma { alpha, beta } => vec![alpha - 1.0, -beta],
        FamilyDef::Beta { alpha, beta } => vec![alpha - 1.0, beta - 1.0],
        FamilyDef::Dirichlet { alpha } => alpha.iter().map(|&a| a - 1.0).collect(),
    }
}

// ───────────────────────────────────────────────────────────────────
// TargetMean extraction
// ───────────────────────────────────────────────────────────────────

/// Extract a scalar target mean.
///
/// # Errors
///
/// Returns `Err` if the target is a vector with length ≠ 1.
pub fn scalar_target(target: &TargetMean, family: &str) -> Result<f64, String> {
    match target {
        TargetMean::Scalar(v) => Ok(*v),
        TargetMean::Vec(v) if v.len() == 1 => v
            .first()
            .copied()
            .ok_or_else(|| format!("{family} target_mean: empty")),
        TargetMean::Vec(v) => Err(format!(
            "{family} target_mean: expected scalar, got {} values",
            v.len()
        )),
    }
}

/// Extract a vector target mean of length `k`.
///
/// # Errors
///
/// Returns `Err` if the vector length doesn't match `k`.
pub fn vec_target(target: &TargetMean, k: usize, family: &str) -> Result<Vec<f64>, String> {
    match target {
        TargetMean::Scalar(v) if k == 2 => Ok(vec![*v, 1.0 - *v]),
        TargetMean::Scalar(_) => Err(format!(
            "{family} target_mean: expected {k} values, got scalar"
        )),
        TargetMean::Vec(v) if v.len() == k => Ok(v.clone()),
        TargetMean::Vec(v) => Err(format!(
            "{family} target_mean: expected {k} values, got {}",
            v.len()
        )),
    }
}
