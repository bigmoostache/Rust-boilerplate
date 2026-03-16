use crate::math::gaussian_pdf;
use crate::model::{NigPosterior, MU_0};

pub fn build_posterior_svg(post: &NigPosterior, obs: &[f64]) -> String {
    let w = 720.0_f64;
    let h = 260.0_f64;
    let pad_l = 10.0_f64;
    let pad_r = 10.0_f64;
    let pad_t = 24.0_f64;
    let pad_b = 22.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;
    let x_min = 35.2_f64;
    let x_max = 40.3_f64;
    let range = x_max - x_min;

    let col_accent = "\x238b2e12";
    let col_accent_semi = "rgba(139,46,18,0.10)";
    let col_accent_ghost = "rgba(139,46,18,0.35)";
    let col_grid = "\x23e8e0d5";
    let col_axis = "\x23c8bfb0";
    let col_prior = "\x23b0a898";
    let col_fever = "\x23c4623e";
    let col_tick_text = "\x238a837a";
    let col_bg = "\x23fafaf8";

    let to_x = |v: f64| -> f64 { pad_l + ((v - x_min) / range) * plot_w };
    let df = post.df();
    let scale = post.scale();
    let peak_student = post.peak_student();
    let y_max = peak_student / 0.80;
    let to_y = |v: f64| -> f64 {
        let ratio = (v / y_max).min(1.05);
        pad_t + plot_h - ratio * plot_h
    };

    let mut svg = String::with_capacity(8192);

    // Background
    svg.push_str(&format!(
        "<svg viewBox=\"0 0 {w} {h}\" xmlns=\"http://www.w3.org/2000/svg\">\
         <rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"{col_bg}\"/>"
    ));

    // Vertical grid lines
    let mut gx = 35.5_f64;
    while gx <= 40.5 {
        let px = to_x(gx);
        svg.push_str(&format!(
            "<line x1=\"{px:.1}\" y1=\"{pad_t}\" x2=\"{px:.1}\" y2=\"{b:.1}\" stroke=\"{col_grid}\" stroke-width=\"1\"/>",
            b = pad_t + plot_h
        ));
        gx += 0.5;
    }

    // X axis
    svg.push_str(&format!(
        "<line x1=\"{pad_l}\" y1=\"{b:.1}\" x2=\"{r:.1}\" y2=\"{b:.1}\" stroke=\"{col_axis}\" stroke-width=\"1\"/>",
        b = pad_t + plot_h,
        r = pad_l + plot_w
    ));

    // Prior (wide Gaussian, scaled for visibility)
    let prior_sigma = 1.0_f64;
    let prior_peak = gaussian_pdf(MU_0, MU_0, prior_sigma);
    let prior_scale_factor = (y_max * 0.35) / prior_peak;
    let mut prior_path = String::new();
    for px in 0..=(plot_w as i32) {
        let xv = x_min + (px as f64 / plot_w) * range;
        let y_val = gaussian_pdf(xv, MU_0, prior_sigma) * prior_scale_factor;
        let ypx = to_y(y_val);
        let xpx = pad_l + px as f64;
        if px == 0 {
            prior_path.push_str(&format!("M{xpx:.1},{ypx:.1}"));
        } else {
            prior_path.push_str(&format!(" L{xpx:.1},{ypx:.1}"));
        }
    }
    svg.push_str(&format!(
        "<path d=\"{prior_path}\" fill=\"none\" stroke=\"{col_prior}\" stroke-width=\"1.5\" stroke-dasharray=\"5,4\"/>"
    ));

    // Student-t fill
    let mut fill_path = String::new();
    fill_path.push_str(&format!("M{pad_l},{b:.1}", b = pad_t + plot_h));
    for px in 0..=(plot_w as i32) {
        let xv = x_min + (px as f64 / plot_w) * range;
        let ypx = to_y(post.student_pdf_at(xv));
        let xpx = pad_l + px as f64;
        fill_path.push_str(&format!(" L{xpx:.1},{ypx:.1}"));
    }
    fill_path.push_str(&format!(
        " L{r:.1},{b:.1} Z",
        r = pad_l + plot_w,
        b = pad_t + plot_h
    ));
    svg.push_str(&format!(
        "<path d=\"{fill_path}\" fill=\"{col_accent_semi}\" stroke=\"none\"/>"
    ));

    // Student-t line (exact marginal)
    let mut student_path = String::new();
    for px in 0..=(plot_w as i32) {
        let xv = x_min + (px as f64 / plot_w) * range;
        let ypx = to_y(post.student_pdf_at(xv));
        let xpx = pad_l + px as f64;
        if px == 0 {
            student_path.push_str(&format!("M{xpx:.1},{ypx:.1}"));
        } else {
            student_path.push_str(&format!(" L{xpx:.1},{ypx:.1}"));
        }
    }
    svg.push_str(&format!(
        "<path d=\"{student_path}\" fill=\"none\" stroke=\"{col_accent}\" stroke-width=\"2.5\"/>"
    ));

    // Gaussian approximation (dashed)
    let mut gauss_path = String::new();
    for px in 0..=(plot_w as i32) {
        let xv = x_min + (px as f64 / plot_w) * range;
        let ypx = to_y(post.gaussian_pdf_at(xv));
        let xpx = pad_l + px as f64;
        if px == 0 {
            gauss_path.push_str(&format!("M{xpx:.1},{ypx:.1}"));
        } else {
            gauss_path.push_str(&format!(" L{xpx:.1},{ypx:.1}"));
        }
    }
    svg.push_str(&format!(
        "<path d=\"{gauss_path}\" fill=\"none\" stroke=\"{col_accent_ghost}\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\"/>"
    ));

    // Fever line at 38°C
    let fx = to_x(38.0);
    svg.push_str(&format!(
        "<line x1=\"{fx:.1}\" y1=\"{pad_t}\" x2=\"{fx:.1}\" y2=\"{b:.1}\" stroke=\"{col_fever}\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\"/>",
        b = pad_t + plot_h
    ));
    svg.push_str(&format!(
        "<text x=\"{tx:.1}\" y=\"{ty:.1}\" fill=\"{col_fever}\" font-size=\"10\" font-weight=\"bold\" font-family=\"JetBrains Mono, monospace\">38\u{00b0}C</text>",
        tx = fx + 4.0, ty = pad_t + 12.0
    ));

    // Observation ticks
    for xi in obs {
        let ox = to_x(*xi);
        svg.push_str(&format!(
            "<line x1=\"{ox:.1}\" y1=\"{y1:.1}\" x2=\"{ox:.1}\" y2=\"{y2:.1}\" stroke=\"{col_accent_ghost}\" stroke-width=\"2\"/>",
            y1 = pad_t + plot_h - 4.0,
            y2 = pad_t + plot_h + 5.0
        ));
    }

    // Mu dot + label
    let mu_x = to_x(post.mu);
    let mu_y = to_y(peak_student);
    svg.push_str(&format!(
        "<circle cx=\"{mu_x:.1}\" cy=\"{mu_y:.1}\" r=\"5\" fill=\"{col_accent}\"/>"
    ));
    let label_x = if mu_x > w * 0.75 {
        mu_x - 130.0
    } else {
        mu_x + 8.0
    };
    svg.push_str(&format!(
        "<text x=\"{label_x:.1}\" y=\"{ly:.1}\" fill=\"{col_accent}\" font-size=\"11\" font-weight=\"bold\" font-family=\"JetBrains Mono, monospace\">\u{03bc} = {mu:.2}\u{00b0}C</text>",
        ly = mu_y + 4.0, mu = post.mu
    ));

    // X labels
    let mut lx = 35.5_f64;
    while lx <= 40.0 {
        let px = to_x(lx);
        svg.push_str(&format!(
            "<text x=\"{px:.1}\" y=\"{ty:.1}\" fill=\"{col_tick_text}\" font-size=\"10\" text-anchor=\"middle\" font-family=\"JetBrains Mono, monospace\">{lx:.1}</text>",
            ty = pad_t + plot_h + pad_b - 4.0
        ));
        lx += 0.5;
    }

    svg.push_str("</svg>");
    svg
}
