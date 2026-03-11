//! Observation models — likelihood terms for measured data.
//!
//! Each observation is produced by an **instrument** that defines its
//! measurement mode and noise parameters.  The observation itself
//! carries only the measured value.
//!
//! The key operation is `E_post[ln p(obs | x)]` — the expected
//! log-likelihood of the observation under the current posterior.

use crate::distributions::NaturalParams;
use crate::distributions::gamma::lgamma;

/// An observation attached to a graph node.
///
/// Each variant corresponds to a **measurement mode** (defined by an
/// instrument).  The noise parameters come from the instrument; the
/// measured value comes from the observation event.
///
/// The observation type must be compatible with the node's distribution
/// family.
#[derive(Debug, Clone)]
pub enum Observation {
    /// Gaussian noise measurement: `p(obs | x) = N(obs; x, σ²_noise)`.
    ///
    /// Compatible with: Gaussian nodes.
    GaussianNoise {
        /// Observed value.
        value: f64,
        /// Known noise variance `σ²_noise > 0` (from instrument).
        noise_var: f64,
    },

    /// Noisy binary channel: `P(obs=1 | p) = (1−ε)·p + ε·(1−p)`.
    ///
    /// Models a binary observation through a symmetric error channel
    /// with flip probability `ε ∈ [0, 0.5)`.
    ///
    /// Compatible with: Bernoulli nodes.
    NoisyChannel {
        /// Observed binary value.
        value: bool,
        /// Symmetric error probability `ε ∈ [0, 0.5)` (from instrument).
        epsilon: f64,
    },

    /// Direct Poisson count observation.
    ///
    /// Compatible with: Poisson nodes.
    PoissonCount {
        /// Observed count.
        count: u64,
    },

    /// Noisy categorical observation with uniform confusion.
    ///
    /// `P(obs=k | x) = (1−ε)·x_k + ε/(K−1)·(1−x_k)`
    /// where `K` is the number of categories and `ε` is the confusion
    /// probability.
    ///
    /// Compatible with: Categorical nodes.
    NoisyCategorical {
        /// Observed category index (0-based).
        category: usize,
        /// Confusion probability `ε ∈ [0, 1)` (from instrument).
        epsilon: f64,
        /// Total number of categories `K` (from node family).
        num_categories: usize,
    },

    /// Beta concentration measurement.
    ///
    /// The observation model: given the true proportion `p` (from a
    /// Beta node), the observed value `v` follows `Beta(κp, κ(1−p))`
    /// where `κ` is the concentration (precision of the measurement).
    ///
    /// Compatible with: Beta nodes.
    BetaConcentration {
        /// Observed proportion in `(0, 1)`.
        value: f64,
        /// Concentration parameter `κ > 0` (from instrument).
        kappa: f64,
    },

    /// Gamma rate measurement.
    ///
    /// The observation model: given the true rate from a Gamma node,
    /// the observed value follows a Gamma distribution with known shape
    /// and rate equal to the node's mean.
    ///
    /// Compatible with: Gamma nodes.
    GammaRate {
        /// Observed positive value.
        value: f64,
        /// Known shape parameter of the observation noise (from instrument).
        shape: f64,
    },

    /// Dirichlet concentration measurement.
    ///
    /// The observation model: given the true proportions `α/Σα` from a
    /// Dirichlet node, the observed proportions follow
    /// `Dir(κ·p₁, κ·p₂, …, κ·pₖ)` where `κ` controls precision.
    ///
    /// Compatible with: Dirichlet nodes.
    DirichletConcentration {
        /// Observed proportions (must sum to ~1).
        values: Vec<f64>,
        /// Concentration parameter `κ > 0` (from instrument).
        kappa: f64,
    },
}

impl Observation {
    /// Expected log-likelihood `E_post[ln p(obs | x)]` under the
    /// posterior distribution `post`.
    ///
    /// This is one of the four ELBO terms. Each (family, obs-type) pair
    /// has its own closed-form expression.
    ///
    /// # Panics
    ///
    /// Panics (debug only) if the observation type is incompatible with
    /// the node family.
    #[must_use]
    pub fn expected_log_likelihood(&self, post: &NaturalParams) -> f64 {
        match (self, post) {
            // ── Gaussian node + Gaussian noise ──────────────────
            (Self::GaussianNoise { value, noise_var }, NaturalParams::Gaussian { eta1, eta2 }) => {
                let sigma2 = -1.0 / (2.0 * eta2);
                let mu = eta1 * sigma2;
                let ex2 = mu.mul_add(mu, sigma2); // E[x²] = μ² + σ²
                let mse = (2.0 * value).mul_add(-mu, value * value) + ex2;
                (-0.5f64).mul_add(
                    (2.0 * std::f64::consts::PI * noise_var).ln(),
                    -(mse / (2.0 * noise_var)),
                )
            }

            // ── Bernoulli node + noisy channel ──────────────────
            (Self::NoisyChannel { value, epsilon }, NaturalParams::Bernoulli { eta1 }) => {
                // P(obs=1|p) = (1−ε)p + ε(1−p) = (1−2ε)p + ε
                // P(obs=0|p) = 1 − P(obs=1|p) = (1−2ε)(1−p) + ε  (by symmetry: swap p↔(1−p))
                //
                // E[ln P(obs|p)] where p ~ Bernoulli posterior with natural param η₁
                // p = sigmoid(η₁)
                let p = sigmoid(*eta1);
                let prob_obs = if *value {
                    (1.0 - 2.0 * epsilon).mul_add(p, *epsilon)
                } else {
                    (1.0 - 2.0 * epsilon).mul_add(1.0 - p, *epsilon)
                };
                // Clamp to avoid log(0)
                prob_obs.max(1e-15).ln()
            }

            // ── Poisson node + Poisson count ────────────────────
            (Self::PoissonCount { count }, NaturalParams::Poisson { eta1 }) => {
                let k = f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
                let ln_k_factorial = lgamma(k + 1.0);
                k.mul_add(*eta1, -eta1.exp()) - ln_k_factorial
            }

            // ── Categorical node + noisy categorical ────────────
            (
                Self::NoisyCategorical {
                    category,
                    epsilon,
                    num_categories,
                },
                NaturalParams::Categorical { eta },
            ) => {
                // P(obs=k|x) = (1−ε)·x_k + ε/(K−1)·(1−x_k)
                //            = (1−ε·K/(K−1))·x_k + ε/(K−1)
                // E[ln P] ≈ ln((1−ε·K/(K−1))·E[x_k] + ε/(K−1))
                let k_f = f64::from(u32::try_from(*num_categories).unwrap_or(u32::MAX));
                let probs = categorical_probs(eta);
                let pk = probs.get(*category).copied().unwrap_or(1.0 / k_f);
                let prob_obs =
                    (1.0 - epsilon * k_f / (k_f - 1.0)).mul_add(pk, epsilon / (k_f - 1.0));
                prob_obs.max(1e-15).ln()
            }

            // ── Beta node + Beta concentration ──────────────────
            (Self::BetaConcentration { value, kappa }, NaturalParams::Beta { eta1, eta2 }) => {
                let alpha = eta1 + 1.0;
                let beta_param = eta2 + 1.0;
                let mean_p = alpha / (alpha + beta_param);
                let a_obs = kappa * mean_p;
                let b_obs = kappa * (1.0 - mean_p);
                (a_obs - 1.0).mul_add(value.ln(), (b_obs - 1.0) * (1.0 - value).ln())
                    - lgamma(a_obs)
                    - lgamma(b_obs)
                    + lgamma(a_obs + b_obs)
            }

            // ── Gamma node + Gamma rate ─────────────────────────
            (Self::GammaRate { value, shape }, NaturalParams::Gamma { eta1, eta2 }) => {
                let alpha = eta1 + 1.0;
                let beta_param = -eta2;
                let mean_rate = alpha / beta_param;
                (shape - 1.0).mul_add(value.ln(), shape * mean_rate.ln())
                    - mean_rate * value
                    - lgamma(*shape)
            }

            // ── Dirichlet node + Dirichlet concentration ────────
            (
                Self::DirichletConcentration { values, kappa },
                NaturalParams::Dirichlet { alpha },
            ) => {
                // Observation model: obs ~ Dir(κ·p) where p = E[x] = α/Σα
                let alpha_sum: f64 = alpha.iter().sum();
                // Σ κ·p_k = κ·Σp_k = κ since p sums to 1
                let mut ll = lgamma(*kappa);
                for (pk, vk) in alpha.iter().zip(values.iter()) {
                    let mean_pk = pk / alpha_sum;
                    let a_k = kappa * mean_pk;
                    ll += (a_k - 1.0) * vk.max(1e-300).ln();
                    ll -= lgamma(a_k);
                }
                ll
            }

            _ => {
                debug_assert!(
                    false,
                    "incompatible observation type for this distribution family"
                );
                f64::NAN
            }
        }
    }
}

/// Compute category probabilities from categorical natural parameters.
fn categorical_probs(eta: &[f64]) -> Vec<f64> {
    let max_eta = eta.iter().copied().fold(0.0_f64, f64::max);
    let mut probs: Vec<f64> = eta.iter().map(|&e| (e - max_eta).exp()).collect();
    probs.push((-max_eta).exp()); // reference class
    let sum: f64 = probs.iter().sum();
    for prob in &mut probs {
        *prob /= sum;
    }
    probs
}

/// Logistic sigmoid: `1 / (1 + exp(−x))`.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_obs_exact() {
        // Posterior N(3, 4) = η₁=0.75, η₂=-0.125. Obs at mean with noise_var=1.
        let post = NaturalParams::Gaussian {
            eta1: 0.75,
            eta2: -0.125,
        };
        let obs = Observation::GaussianNoise {
            value: 3.0,
            noise_var: 1.0,
        };
        let ll = obs.expected_log_likelihood(&post);
        let expected = (-0.5f64).mul_add((2.0 * std::f64::consts::PI).ln(), -4.0 / 2.0);
        assert!(
            (ll - expected).abs() < 1e-10,
            "ll={ll}, expected={expected}"
        );
    }

    #[test]
    fn noisy_channel_no_noise() {
        // ε=0 → same as exact Bernoulli: P(obs=1|p) = p
        let post = NaturalParams::Bernoulli {
            eta1: (0.7_f64 / 0.3).ln(),
        };
        let obs_true = Observation::NoisyChannel {
            value: true,
            epsilon: 0.0,
        };
        let ll = obs_true.expected_log_likelihood(&post);
        assert!((ll - 0.7_f64.ln()).abs() < 1e-10);

        let obs_false = Observation::NoisyChannel {
            value: false,
            epsilon: 0.0,
        };
        let ll_false = obs_false.expected_log_likelihood(&post);
        assert!((ll_false - 0.3_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn noisy_channel_with_noise() {
        // ε=0.1, p=0.7 → P(obs=1|p) = 0.8*0.7 + 0.1 = 0.66
        let post = NaturalParams::Bernoulli {
            eta1: (0.7_f64 / 0.3).ln(),
        };
        let obs = Observation::NoisyChannel {
            value: true,
            epsilon: 0.1,
        };
        let ll = obs.expected_log_likelihood(&post);
        let expected = 0.66_f64.ln();
        assert!(
            (ll - expected).abs() < 1e-10,
            "ll={ll}, expected={expected}"
        );
    }

    #[test]
    fn noisy_categorical_no_noise() {
        // ε=0 → P(obs=k|x) = x_k
        let post = NaturalParams::Categorical {
            eta: vec![(0.2_f64 / 0.5).ln(), (0.3_f64 / 0.5).ln()],
        };
        let obs = Observation::NoisyCategorical {
            category: 0,
            epsilon: 0.0,
            num_categories: 3,
        };
        let ll = obs.expected_log_likelihood(&post);
        assert!((ll - 0.2_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn poisson_obs() {
        let post = NaturalParams::Poisson { eta1: 5.0_f64.ln() };
        let obs = Observation::PoissonCount { count: 3 };
        let ll = obs.expected_log_likelihood(&post);
        let expected = 3.0f64.mul_add(5.0_f64.ln(), -5.0) - lgamma(4.0);
        assert!((ll - expected).abs() < 1e-10);
    }
}
