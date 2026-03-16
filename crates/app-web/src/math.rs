pub fn log_gamma(z: f64) -> f64 {
    let c: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if z < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln()
            - log_gamma(1.0 - z);
    }
    let zz = z - 1.0;
    let mut x = c[0];
    for (i, coeff) in c.iter().enumerate().skip(1) {
        x += coeff / (zz + i as f64);
    }
    let t = zz + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (zz + 0.5) * t.ln() - t + x.ln()
}

pub fn student_pdf(x: f64, mu: f64, scale: f64, df: f64) -> f64 {
    let z = (x - mu) / scale;
    let log_norm = log_gamma((df + 1.0) / 2.0)
        - log_gamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - scale.ln();
    let log_kernel = -((df + 1.0) / 2.0) * (1.0 + z * z / df).ln();
    (log_norm + log_kernel).exp()
}

pub fn gaussian_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}
