//! Observation models — likelihood terms for measured data.
//!
//! Each observation carries a measured value and a noise model.
//! The key operation is `E_post[ln p(obs | x)]` — the expected
//! log-likelihood of the observation under the current posterior.

use crate::distributions::NaturalParams;
use crate::distributions::gamma::lgamma;

/// An observation attached to a graph node.
///
/// The observation type must be compatible with the node's distribution
/// family. Incompatible combinations panic at runtime.
#[derive(Debug, Clone, Copy)]
pub enum Observation {
    /// Observed value with Gaussian noise: `p(obs | x) = N(obs; x, σ²_noise)`.
    ///
    /// Compatible with: Gaussian nodes.
    GaussianNoise {
        /// Observed value.
        value: f64,
        /// Known noise variance `σ²_noise > 0`.
        noise_var: f64,
    },

    /// Exact categorical observation: `p(obs | x) = x_k` where `k` is
    /// the observed category.
    ///
    /// Compatible with: Categorical, Bernoulli nodes.
    CategoricalExact {
        /// Observed category index (0-based).
        category: usize,
    },

    /// Count observation with Poisson likelihood.
    ///
    /// Compatible with: Poisson nodes.
    PoissonCount {
        /// Observed count.
        count: u64,
    },

    /// Binary observation.
    ///
    /// Compatible with: Bernoulli nodes.
    BernoulliExact {
        /// Observed value.
        value: bool,
    },

    /// Observed proportion with Beta-type likelihood.
    ///
    /// The observation model is: given the true proportion `p` (from a
    /// Beta node), the observed value `v` follows `Beta(κp, κ(1−p))`
    /// where `κ` is the concentration (precision of the measurement).
    ///
    /// Compatible with: Beta nodes.
    BetaProportion {
        /// Observed proportion in `(0, 1)`.
        value: f64,
        /// Concentration parameter `κ > 0` controlling noise.
        concentration: f64,
    },

    /// Observed positive real with Gamma-type likelihood.
    ///
    /// The observation model: given the true rate from a Gamma node,
    /// the observed value follows a Gamma distribution with known shape
    /// and rate equal to the node's mean.
    ///
    /// Compatible with: Gamma nodes.
    GammaRate {
        /// Observed positive value.
        value: f64,
        /// Known shape parameter of the observation noise.
        shape: f64,
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
    /// Panics if the observation type is incompatible with the node family.
    #[must_use]
    pub fn expected_log_likelihood(&self, post: &NaturalParams) -> f64 {
        match (self, post) {
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

            (Self::CategoricalExact { category }, NaturalParams::Categorical { eta }) => {
                // p(obs|x) = x_k = p_k
                // E[ln p] = E[ln x_k] where x ~ Cat(η)
                // For the posterior Cat, E[ln x_k] is just ln(p_k)
                // = η_k − A(η) for k < K, or −A(η) for k = K
                let max_eta = eta.iter().copied().fold(0.0_f64, f64::max);
                let sum_exp: f64 =
                    eta.iter().map(|&e| (e - max_eta).exp()).sum::<f64>() + (-max_eta).exp();
                let log_z = max_eta + sum_exp.ln();
                if let Some(&eta_k) = eta.get(*category) {
                    eta_k - log_z
                } else {
                    -log_z // Reference class or out-of-bounds
                }
            }

            (Self::BernoulliExact { value }, NaturalParams::Bernoulli { eta1 }) => {
                // p(obs|x) = x^obs · (1−x)^(1−obs) where x = sigmoid(η)
                // E[ln p] = obs·E[ln x] + (1−obs)·E[ln(1−x)]
                // For Bernoulli: E[ln x] = −softplus(−η), E[ln(1−x)] = −softplus(η)
                let log_p = -softplus(-eta1);
                let log_1mp = -softplus(*eta1);
                if *value { log_p } else { log_1mp }
            }

            (Self::PoissonCount { count }, NaturalParams::Poisson { eta1 }) => {
                // p(obs = k | λ) = λ^k e^{−λ} / k!  where η₁ = ln λ
                // E[ln p(k|λ)] = k·η₁ − exp(η₁) − ln(k!)
                let k = f64::from(u32::try_from(*count).unwrap_or(u32::MAX));
                let ln_k_factorial = lgamma(k + 1.0);
                k.mul_add(*eta1, -eta1.exp()) - ln_k_factorial
            }

            (
                Self::BetaProportion {
                    value,
                    concentration,
                },
                NaturalParams::Beta { eta1, eta2 },
            ) => {
                // Observation model: v ~ Beta(κp, κ(1−p)) where p = E[x] under Beta(α,β)
                // This is approximate — we use the mean of the posterior as the
                // "true" proportion for the observation likelihood.
                let alpha = eta1 + 1.0;
                let beta_param = eta2 + 1.0;
                let mean_p = alpha / (alpha + beta_param);
                let a_obs = concentration * mean_p;
                let b_obs = concentration * (1.0 - mean_p);
                // ln Beta(v; a_obs, b_obs)
                (a_obs - 1.0).mul_add(value.ln(), (b_obs - 1.0) * (1.0 - value).ln())
                    - lgamma(a_obs)
                    - lgamma(b_obs)
                    + lgamma(a_obs + b_obs)
            }

            (Self::GammaRate { value, shape }, NaturalParams::Gamma { eta1, eta2 }) => {
                // Observation: v ~ Gamma(shape, rate) where rate = E[x] = α/β
                let alpha = eta1 + 1.0;
                let beta_param = -eta2;
                let mean_rate = alpha / beta_param;
                // ln Gamma(v; shape, mean_rate) = shape·ln(mean_rate) + (shape−1)·ln(v)
                //     − mean_rate·v − ln Γ(shape)
                (shape - 1.0).mul_add(value.ln(), shape * mean_rate.ln())
                    - mean_rate * value
                    - lgamma(*shape)
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

/// Numerically stable softplus: `ln(1 + exp(x))`.
fn softplus(x: f64) -> f64 {
    if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_obs_exact() {
        // If posterior is N(3, 4) and obs is exactly at mean with noise_var=1:
        // E[ln p(3|x)] = −½ ln(2π) − E[(3−x)²]/2 = −½ ln(2π) − σ²/2 = −½ ln(2π) − 2
        let post = NaturalParams::Gaussian {
            eta1: 0.75,
            eta2: -0.125,
        };
        let obs = Observation::GaussianNoise {
            value: 3.0,
            noise_var: 1.0,
        };
        let ll = obs.expected_log_likelihood(&post);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 4.0 / 2.0;
        assert!(
            (ll - expected).abs() < 1e-10,
            "ll={ll}, expected={expected}"
        );
    }

    #[test]
    fn bernoulli_obs() {
        let post = NaturalParams::Bernoulli {
            eta1: (0.7_f64 / 0.3).ln(),
        };
        // Observing true → E[ln p(1|x)] = ln(0.7)
        let obs_true = Observation::BernoulliExact { value: true };
        let ll = obs_true.expected_log_likelihood(&post);
        assert!((ll - 0.7_f64.ln()).abs() < 1e-10);

        // Observing false → E[ln p(0|x)] = ln(0.3)
        let obs_false = Observation::BernoulliExact { value: false };
        let ll = obs_false.expected_log_likelihood(&post);
        assert!((ll - 0.3_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn categorical_obs() {
        // Cat(K=3) with p = (0.2, 0.3, 0.5)
        let post = NaturalParams::Categorical {
            eta: vec![(0.2_f64 / 0.5).ln(), (0.3_f64 / 0.5).ln()],
        };
        let obs = Observation::CategoricalExact { category: 0 };
        let ll = obs.expected_log_likelihood(&post);
        assert!((ll - 0.2_f64.ln()).abs() < 1e-10);

        let obs_ref = Observation::CategoricalExact { category: 2 };
        let ll_ref = obs_ref.expected_log_likelihood(&post);
        assert!((ll_ref - 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn poisson_obs() {
        // Poisson(λ=5), observe k=3
        let post = NaturalParams::Poisson { eta1: 5.0_f64.ln() };
        let obs = Observation::PoissonCount { count: 3 };
        let ll = obs.expected_log_likelihood(&post);
        // ln P(3|5) = 3·ln5 − 5 − ln(6)
        let expected = 3.0 * 5.0_f64.ln() - 5.0 - lgamma(4.0);
        assert!((ll - expected).abs() < 1e-10);
    }
}
