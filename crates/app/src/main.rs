//! Quick test script for the exponential family crate.

use std::io::Write as _;
use std::time::Instant;

use exponential::distribution::ExponentialFamily as _;
use exponential::inference::ConjugatePrior;
use exponential::laws::gaussian::Normal1D;
use nalgebra::DVector;

/// Write a line to stdout, ignoring IO errors (test script).
fn w(out: &mut std::io::Stdout, msg: &str) {
    match out.write_all(msg.as_bytes()) {
        Ok(()) | Err(_) => {}
    }
    match out.write_all(b"\n") {
        Ok(()) | Err(_) => {}
    }
}

/// Format and write a line to stdout.
macro_rules! wf {
    ($out:expr, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        let mut buf = String::new();
        let _ignored = write!(buf, $($arg)*);
        w($out, &buf);
    }};
}

/// Get element `i` from a `DVector`, returning 0.0 on OOB.
fn get(v: &DVector<f64>, i: usize) -> f64 {
    v.get(i).copied().unwrap_or(0.0)
}

fn main() {
    let start = Instant::now();
    let out = &mut std::io::stdout();

    // ── 1. Create N(μ=37, σ²=0.5) and inspect ─────────────────────────
    let normal = match Normal1D::new(37.0, 0.5) {
        Ok(n) => n,
        Err(_e) => return,
    };
    wf!(
        out,
        "=== 1D Normal: μ={}, σ²={}, σ={:.4} ===",
        normal.mu(),
        normal.sigma_sq(),
        normal.sigma()
    );

    // ── 2. Standard ↔ Natural roundtrip ─────────────────────────────
    let eta = normal.to_natural();
    wf!(
        out,
        "\nto_natural:   η = [{:.6}, {:.6}]",
        get(&eta, 0),
        get(&eta, 1)
    );

    let recovered = match Normal1D::from_natural(&eta) {
        Ok(r) => r,
        Err(_e) => return,
    };
    wf!(
        out,
        "from_natural: μ={}, σ²={}",
        recovered.mu(),
        recovered.sigma_sq()
    );

    // ── 3. Sufficient statistic T(x) = (x, x²) ────────────────────
    let x = 3.0;
    let t = match normal.sufficient_statistic(&x) {
        Ok(v) => v,
        Err(_e) => return,
    };
    wf!(out, "\nT({}) = [{}, {}]", x, get(&t, 0), get(&t, 1));

    // ── 4. Log-partition A(η) and gradient ∇A(η) = (E[x], E[x²]) ──
    let a = match normal.log_partition(&eta) {
        Ok(v) => v,
        Err(_e) => return,
    };
    let grad = match normal.log_partition_gradient(&eta) {
        Ok(v) => v,
        Err(_e) => return,
    };
    wf!(out, "A(η)  = {:.6}", a);
    wf!(
        out,
        "∇A(η) = [{:.6}, {:.6}]  (expect μ=37, μ²+σ²=1370.5)",
        get(&grad, 0),
        get(&grad, 1)
    );

    // ── 5. Log-density at a few points ──────────────────────────────
    w(out, "\nlog p(x | μ=5, σ²=2):");
    for xi in [3.0, 4.0, 5.0, 6.0, 7.0] {
        if let Ok(lp) = normal.log_density(&xi, &eta) {
            wf!(out, "  x={:.1}  →  ln p = {:.6}", xi, lp);
        }
    }

    // ── 6. Conjugate prior + Bayesian update ────────────────────────
    let lambda0 = DVector::from_vec(vec![0.0, 0.0]);
    let nu0 = 1.0;
    let prior = match ConjugatePrior::new(lambda0, nu0) {
        Ok(p) => p,
        Err(_e) => return,
    };
    w(out, "\n=== Conjugate Prior ===");
    wf!(
        out,
        "λ₀ = [{}, {}], ν₀ = {}",
        get(prior.lambda(), 0),
        get(prior.lambda(), 1),
        prior.nu()
    );
    let pm = prior.prior_mean();
    wf!(out, "prior mean E[T] = [{}, {}]", get(&pm, 0), get(&pm, 1));

    let data = [4.5, 5.1, 5.3, 4.8, 5.0, 5.2, 4.9, 5.1, 5.4, 4.7];
    let posterior = match prior.update_batch(&normal, &data) {
        Ok(p) => p,
        Err(_e) => return,
    };
    wf!(out, "\nAfter {} observations:", data.len());
    wf!(
        out,
        "λₙ  = [{}, {}]",
        get(posterior.lambda(), 0),
        get(posterior.lambda(), 1)
    );
    wf!(out, "νₙ  = {}", posterior.nu());
    let post_mean = posterior.posterior_mean();
    wf!(
        out,
        "posterior mean E[T] = [{:.6}, {:.6}]",
        get(&post_mean, 0),
        get(&post_mean, 1)
    );
    if let Ok(s) = posterior.shrinkage_weight(data.len()) {
        wf!(out, "shrinkage toward prior = {:.4}", s);
    }

    // ── 7. Sequential vs batch update (should be identical) ─────────
    let mut sequential = prior.clone();
    let mut ok = true;
    for xi in &data {
        match sequential.update_single(&normal, xi) {
            Ok(s) => sequential = s,
            Err(_e) => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        let batch_mean = posterior.posterior_mean();
        let seq_mean = sequential.posterior_mean();
        let diff: f64 = batch_mean
            .zip_map(&seq_mean, |lhs: f64, rhs: f64| lhs - rhs)
            .norm();
        wf!(
            out,
            "\nsequential vs batch difference: {:.2e} (should be ~0)",
            diff
        );
    }

    // ── 8. Prior unnormalized log-density at η ──────────────────────
    if let Ok(prior_lp) = prior.unnormalized_log_density(&normal, &eta) {
        wf!(out, "\nunnorm log prior(η)     = {:.6}", prior_lp);
    }
    if let Ok(post_lp) = posterior.unnormalized_log_density(&normal, &eta) {
        wf!(out, "unnorm log posterior(η) = {:.6}", post_lp);
    }

    // ── Timing ──────────────────────────────────────────────────────
    let secs = start.elapsed().as_secs_f64();
    wf!(out, "\n⏱ total: {:.6}s", secs);
}
