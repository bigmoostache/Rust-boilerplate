use crate::math::{gaussian_pdf, student_pdf};

pub const MU_0: f64 = 36.2;
pub const NU_0: f64 = 0.001;
pub const ALPHA_0: f64 = 1.0;
pub const BETA_0: f64 = 0.0625;

#[derive(Debug, Clone)]
pub struct NigPosterior {
    pub mu: f64,
    pub nu: f64,
    pub alpha: f64,
    pub beta: f64,
    pub lambda1: f64,
    pub lambda2: f64,
}

impl NigPosterior {
    pub fn df(&self) -> f64 {
        2.0 * self.alpha
    }

    pub fn scale(&self) -> f64 {
        (self.beta / (self.alpha * self.nu)).sqrt()
    }

    pub fn student_pdf_at(&self, x: f64) -> f64 {
        student_pdf(x, self.mu, self.scale(), self.df())
    }

    pub fn gaussian_pdf_at(&self, x: f64) -> f64 {
        let sigma = self.scale();
        gaussian_pdf(x, self.mu, sigma)
    }

    pub fn peak_student(&self) -> f64 {
        self.student_pdf_at(self.mu)
    }
}

pub fn compute_posterior(obs: &[f64]) -> NigPosterior {
    let n = obs.len() as f64;
    if n == 0.0 {
        return NigPosterior {
            mu: MU_0,
            nu: NU_0,
            alpha: ALPHA_0,
            beta: BETA_0,
            lambda1: 0.0,
            lambda2: 0.0,
        };
    }
    let sum_x: f64 = obs.iter().copied().sum();
    let xbar = sum_x / n;
    let s: f64 = obs.iter().map(|x| (x - xbar) * (x - xbar)).sum();

    let nu_n = NU_0 + n;
    let mu_n = (NU_0 * MU_0 + sum_x) / nu_n;
    let alpha_n = ALPHA_0 + n / 2.0;
    let beta_n = BETA_0 + 0.5 * s + (NU_0 * n * (xbar - MU_0) * (xbar - MU_0)) / (2.0 * nu_n);

    let lambda1 = NU_0 * MU_0 / BETA_0 + sum_x;
    let sum_x2: f64 = obs.iter().map(|x| x * x).sum();
    let lambda2 = -(NU_0 / (2.0 * BETA_0) + sum_x2);

    NigPosterior {
        mu: mu_n,
        nu: nu_n,
        alpha: alpha_n,
        beta: beta_n,
        lambda1,
        lambda2,
    }
}
