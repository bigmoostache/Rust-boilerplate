use dioxus::prelude::*;
use exponential::inference::ConjugatePrior;
use exponential::laws::gaussian::Normal1D;
use nalgebra::DVector;

const STYLE: Asset = asset!("/assets/style.css");

// ─── Model ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct Measurement {
    id: u64,
    value: f64,
}

#[derive(Debug, Clone)]
struct Model {
    measurements: Vec<Measurement>,
    next_id: u64,
    nu0: f64,
    sigma_thermometer: f64,
}

impl Model {
    fn new() -> Self {
        Self {
            measurements: Vec::new(),
            next_id: 1,
            nu0: 0.001,
            sigma_thermometer: 0.5,
        }
    }

    fn sigma_sq(&self) -> f64 {
        self.sigma_thermometer * self.sigma_thermometer
    }

    fn prior(&self) -> ConjugatePrior {
        let mu0 = 37.0;
        // Prior uses a fixed reference variance so that changing the
        // thermometer slider doesn't alter the prior when there is no data.
        let prior_sigma_sq = 0.25;
        let lambda0 = DVector::from_vec(vec![
            self.nu0 * mu0,
            self.nu0 * (mu0 * mu0 + prior_sigma_sq),
        ]);
        ConjugatePrior::new(lambda0, self.nu0).expect("valid prior")
    }

    fn likelihood(&self) -> Normal1D {
        Normal1D::new(0.0, self.sigma_sq()).expect("valid variance")
    }

    fn posterior(&self) -> ConjugatePrior {
        let prior = self.prior();
        let likelihood = self.likelihood();
        let data: Vec<f64> = self.measurements.iter().map(|m| m.value).collect();
        if data.is_empty() {
            prior
        } else {
            prior
                .update_batch(&likelihood, &data)
                .expect("valid update")
        }
    }

    fn posterior_mean(&self) -> f64 {
        self.posterior().posterior_mean()[0]
    }

    fn posterior_std(&self) -> f64 {
        let nu = self.posterior().nu();
        (self.sigma_sq() / nu).sqrt()
    }

    fn credible_interval(&self) -> (f64, f64) {
        let m = self.posterior_mean();
        let s = self.posterior_std();
        (m - 1.96 * s, m + 1.96 * s)
    }

    fn add(&mut self, value: f64) {
        let id = self.next_id;
        self.next_id += 1;
        self.measurements.push(Measurement { id, value });
    }

    fn remove(&mut self, id: u64) {
        self.measurements.retain(|m| m.id != id);
    }
}

// ─── Bell curve helpers ─────────────────────────────────────────

fn bell_curve_svg(mean: f64, std: f64, measurements: &[Measurement]) -> String {
    let col_accent = "\x232563eb";
    let col_fill = "\x23eff3fe";
    let col_tick = "\x238b95a5";
    let col_grid = "\x23e2e4e9";

    let effective_std = if std > 5.0 { 5.0 } else { std };
    let lo = mean - 4.0 * effective_std;
    let hi = mean + 4.0 * effective_std;
    let w = 720.0_f64;
    let h = 320.0_f64;
    let pad_top = 24.0_f64;
    let pad_bot = 36.0_f64;
    let plot_h = h - pad_top - pad_bot;
    let base = h - pad_bot;

    let n_points = 200;
    let peak = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());

    let mut path = String::with_capacity(2048);
    for i in 0..=n_points {
        let frac = i as f64 / n_points as f64;
        let x_val = lo + frac * (hi - lo);
        let px = frac * w;
        let z = (x_val - mean) / std;
        let pdf = (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt());
        let py = pad_top + plot_h * (1.0 - pdf / peak);
        if i == 0 {
            path.push_str(&format!("M{px:.1},{py:.1}"));
        } else {
            path.push_str(&format!(" L{px:.1},{py:.1}"));
        }
    }

    let fill_path = format!("{path} L{w:.1},{base:.1} L0,{base:.1} Z");

    // Ticks
    let mut ticks = String::new();
    let tick_step = if effective_std > 2.0 {
        2.0
    } else if effective_std > 0.5 {
        0.5
    } else {
        0.1
    };
    let first_tick = (lo / tick_step).ceil() * tick_step;
    let mut t = first_tick;
    while t <= hi {
        let px = ((t - lo) / (hi - lo)) * w;
        let y1 = base;
        let y2 = y1 + 6.0;
        let ty = y2 + 14.0;
        ticks.push_str(&format!(
            "<line x1=\"{px:.1}\" y1=\"{y1:.1}\" x2=\"{px:.1}\" y2=\"{y2:.1}\" stroke=\"{col_tick}\" stroke-width=\"1\"/>"
        ));
        ticks.push_str(&format!(
            "<text x=\"{px:.1}\" y=\"{ty:.1}\" fill=\"{col_tick}\" font-size=\"12\" text-anchor=\"middle\" font-family=\"Inter, sans-serif\">{t:.1}</text>"
        ));
        t += tick_step;
    }

    // Data points
    let mut dots = String::new();
    for m in measurements {
        let frac = (m.value - lo) / (hi - lo);
        if (0.0..=1.0).contains(&frac) {
            let px = frac * w;
            dots.push_str(&format!(
                "<circle cx=\"{px:.1}\" cy=\"{base:.1}\" r=\"5.5\" fill=\"{col_accent}\" stroke=\"white\" stroke-width=\"2\"/>"
            ));
        }
    }

    // Mean line
    let mean_px = ((mean - lo) / (hi - lo)) * w;

    format!(
        "<svg viewBox=\"0 0 {w} {h}\" xmlns=\"http://www.w3.org/2000/svg\">\
         <path d=\"{fill_path}\" fill=\"{col_fill}\" stroke=\"none\"/>\
         <path d=\"{path}\" fill=\"none\" stroke=\"{col_accent}\" stroke-width=\"2.5\"/>\
         <line x1=\"0\" y1=\"{base:.1}\" x2=\"{w}\" y2=\"{base:.1}\" stroke=\"{col_grid}\" stroke-width=\"1\"/>\
         <line x1=\"{mean_px:.1}\" y1=\"{pad_top:.1}\" x2=\"{mean_px:.1}\" y2=\"{base:.1}\" stroke=\"{col_accent}\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\"/>\
         {ticks}\
         {dots}\
         </svg>"
    )
}

// ─── App ────────────────────────────────────────────────────────

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let model = use_signal(Model::new);

    rsx! {
        document::Stylesheet { href: STYLE }
        div { class: "app-shell",
            header { class: "app-header",
                div { class: "header-inner",
                    h1 { "Body Temperature" }
                    span { class: "subtitle", "Bayesian inference" }
                }
            }
            main { class: "app-main",
                PosteriorCard { model }
                ConfigCard { model }
                InputCard { model }
                ListCard { model }
            }
        }
    }
}

// ─── Posterior Card ──────────────────────────────────────────────

#[component]
fn PosteriorCard(model: Signal<Model>) -> Element {
    let m = model();
    let mean = m.posterior_mean();
    let std = m.posterior_std();
    let (lo, hi) = m.credible_interval();
    let nu = m.posterior().nu();
    let count = m.measurements.len();
    let svg_html = bell_curve_svg(mean, std, &m.measurements);

    rsx! {
        div { class: "card",
            div { class: "card-header",
                h2 { "Posterior distribution" }
                span { class: "badge", "{count} measurement(s)" }
            }
            div { class: "card-body",
                div { class: "bell-curve-container",
                    div { dangerous_inner_html: "{svg_html}" }
                }
                div { class: "stat-grid",
                    div { class: "stat-card",
                        div { class: "stat-label", "Estimated temperature" }
                        div { class: "stat-value",
                            "{mean:.2}"
                            span { class: "stat-unit", "°C" }
                        }
                    }
                    div { class: "stat-card",
                        div { class: "stat-label", "Uncertainty (σ)" }
                        div { class: "stat-value",
                            "{std:.3}"
                            span { class: "stat-unit", "°C" }
                        }
                    }
                    div { class: "stat-card",
                        div { class: "stat-label", "95% credible interval" }
                        div { class: "stat-value",
                            "[{lo:.2}, {hi:.2}]"
                            span { class: "stat-unit", "°C" }
                        }
                    }
                }
            }
            div { class: "stat-footer",
                {
                    let sigma_sq_over_nu = m.sigma_sq() / nu;
                    format!("ν = {nu:.4}  ·  σ²/ν = {sigma_sq_over_nu:.4}")
                }
            }
        }
    }
}

// ─── Configuration Card ─────────────────────────────────────────

#[component]
fn ConfigCard(model: Signal<Model>) -> Element {
    let nu0 = model().nu0;
    let sigma = model().sigma_thermometer;

    // For ν₀ slider: log scale from 0.0001 to 100
    let nu0_log = nu0.ln();
    let nu0_min = 0.0001_f64.ln();
    let nu0_max = 100.0_f64.ln();

    rsx! {
        div { class: "card",
            div { class: "card-header",
                h2 { "Configuration" }
            }
            div { class: "card-body",
                div { class: "config-grid",
                    div { class: "slider-group",
                        div { class: "slider-label",
                            span { class: "slider-label-text", "Prior strength (ν₀)" }
                            span { class: "slider-label-value", "{nu0:.4}" }
                        }
                        input {
                            r#type: "range",
                            min: "{nu0_min}",
                            max: "{nu0_max}",
                            step: "0.01",
                            value: "{nu0_log}",
                            oninput: move |e| {
                                if let Ok(v) = e.value().parse::<f64>() {
                                    model.write().nu0 = v.exp();
                                }
                            },
                        }
                        span { class: "slider-hint",
                            "Low = weak prior (data dominates). High = strong prior (prior dominates)."
                        }
                    }
                    div { class: "slider-group",
                        div { class: "slider-label",
                            span { class: "slider-label-text", "Thermometer uncertainty (σ)" }
                            span { class: "slider-label-value", "{sigma:.2} °C" }
                        }
                        input {
                            r#type: "range",
                            min: "0.05",
                            max: "3.0",
                            step: "0.05",
                            value: "{sigma}",
                            oninput: move |e| {
                                if let Ok(v) = e.value().parse::<f64>() {
                                    model.write().sigma_thermometer = v;
                                }
                            },
                        }
                        span { class: "slider-hint",
                            "Standard deviation of the thermometer reading error in °C."
                        }
                    }
                }
            }
        }
    }
}

// ─── Input Card ─────────────────────────────────────────────────

#[component]
fn InputCard(model: Signal<Model>) -> Element {
    let mut input_val = use_signal(String::new);

    let mut submit = move || {
        if let Ok(v) = input_val().parse::<f64>() {
            model.write().add(v);
            input_val.set(String::new());
        }
    };

    rsx! {
        div { class: "card",
            div { class: "card-header",
                h2 { "Add measurement" }
            }
            div { class: "card-body",
                div { class: "input-row",
                    input {
                        class: "input-field",
                        r#type: "number",
                        step: "0.1",
                        placeholder: "e.g. 37.2",
                        value: "{input_val}",
                        oninput: move |e| input_val.set(e.value()),
                        onkeydown: move |e| {
                            if e.key() == Key::Enter {
                                submit();
                            }
                        },
                    }
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| submit(),
                        "Add"
                    }
                }
                p { class: "input-hint", "Temperature in °C. Press Enter or click Add." }
            }
        }
    }
}

// ─── Measurement List Card ──────────────────────────────────────

#[component]
fn ListCard(model: Signal<Model>) -> Element {
    let measurements = model().measurements.clone();
    let count = measurements.len();

    rsx! {
        div { class: "card",
            div { class: "card-header",
                h2 { "Measurements" }
                span { class: "badge", "{count}" }
            }
            div { class: "card-body",
                if measurements.is_empty() {
                    p { class: "empty-state", "No measurements yet." }
                } else {
                    ul { class: "measurement-list",
                        for m in measurements {
                            li { key: "{m.id}",
                                class: "measurement-item",
                                div { class: "measurement-left",
                                    span { class: "measurement-value", "{m.value:.2} °C" }
                                    span { class: "measurement-meta", "#{m.id}" }
                                }
                                button {
                                    class: "btn btn-danger-ghost",
                                    onclick: move |_| model.write().remove(m.id),
                                    "Remove"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
