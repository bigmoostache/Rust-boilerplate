//! Coupling calibration — analytical resolution.
//!
//! Given two nodes `A` (family `F_A`) and `B` (family `F_B`) coupled
//! by a matrix `B ∈ ℝ^{d_A × d_B}`, calibration determines the
//! entries of `B` from user-provided statements such as:
//!
//! - `E[X_A | X_B = v] = target`  (conditional mean)
//! - `V[X_A | X_B = v] = V[X_A]` (variance unchanged)
//! - `B · T_B(v) = 0`            (no effect at a point)
//! - `B[i,j] = 0`                (hard zero on one entry)
//!
//! The key identity is:
//!
//! ```text
//! η_A^cond(x_B) = η_A + B · T_B(x_B)
//! ```
//!
//! so the conditional `p(x_A | x_B)` stays in family `F_A` with a
//! shifted natural parameter.  Most constraints are **linear** in
//! the entries of `B`, provided the "shape" (variance, concentration)
//! is held fixed.

pub mod parse;
pub mod suff_stats;
pub mod system;

#[cfg(test)]
mod tests;
