//! Global constants for the inference engine.
//!
//! All tunable numerical constants live here so they can be reviewed
//! and adjusted in one place.

// ---------------------------------------------------------------------------
// Inference — damping and convergence
// ---------------------------------------------------------------------------

/// Minimum damping coefficient `α_i`.
///
/// Prevents nodes with very large Fisher information from stalling
/// (α → 0).  The Jacobian-based formula `α_i = 1/max(1, Σ||J_ij||_2)`
/// can be overly conservative for Gaussian nodes with large variance
/// (Fisher entries scale as `σ⁴`), leading to `α ≈ 0` and thousands
/// of iterations.  This floor ensures practical convergence.
///
/// **Trade-off**: values above the theoretical `α` may cause mild
/// oscillation in strongly coupled graphs, but the clamping in
/// [`super::inference::clamp_natural_params`] prevents divergence.
pub const MIN_ALPHA: f64 = 0.05;

/// Minimum distance from the domain boundary when clamping natural
/// parameters.
///
/// Prevents degenerate distributions (e.g. Gaussian with σ² = 0).
pub const NATURAL_PARAM_EPS: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Special functions — Bernoulli numbers and asymptotic coefficients
// ---------------------------------------------------------------------------

/// Even Bernoulli numbers `B₂, B₄, B₆, B₈, B₁₀` used in the
/// asymptotic expansions of digamma `ψ(z)` and trigamma `ψ'(z)`.
///
/// Reference: <https://en.wikipedia.org/wiki/Bernoulli_number>
/// `B₂ = 1/6`.
pub const B2: f64 = 1.0 / 6.0;
/// `B₄ = −1/30`.
pub const B4: f64 = -1.0 / 30.0;
/// `B₆ = 1/42`.
pub const B6: f64 = 1.0 / 42.0;
/// `B₈ = −1/30`.
pub const B8: f64 = -1.0 / 30.0;
/// `B₁₀ = 5/66`.
pub const B10: f64 = 5.0 / 66.0;

/// Digamma asymptotic coefficients: `−B_{2k} / (2k)` for `k = 1..5`.
///
/// ```text
/// ψ(z) = ln(z) − 1/(2z) − Σ_k B_{2k} / (2k · z^{2k})
/// ```
///
/// These are the negated coefficients multiplied into `z⁻²ᵏ`:
/// `DIGAMMA_ASYM[i] = B_{2(i+1)} / (2(i+1))`.
pub const DIGAMMA_ASYM: [f64; 5] = [
    B2 / 2.0,   // B₂/2  =  1/12
    B4 / 4.0,   // B₄/4  = -1/120
    B6 / 6.0,   // B₆/6  =  1/252
    B8 / 8.0,   // B₈/8  = -1/240
    B10 / 10.0, // B₁₀/10 =  1/132
];

/// Trigamma asymptotic coefficients: `B_{2k}` for `k = 1..5`.
///
/// ```text
/// ψ'(z) = 1/z + 1/(2z²) + Σ_k B_{2k} / z^{2k+1}
/// ```
///
/// These are multiplied into `z⁻(2k+1)`.
pub const TRIGAMMA_ASYM: [f64; 5] = [
    B2,  //  1/6
    B4,  // -1/30
    B6,  //  1/42
    B8,  // -1/30
    B10, //  5/66
];
