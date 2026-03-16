//! Body temperature Bayesian inference webapp.
//!
//! Single-page app at `/body-temperature` that:
//! - Shows posterior distribution for true body temperature θ
//! - Allows CRUD of thermometer measurements
//! - Updates posterior live as measurements are added/removed
//!
//! Model:
//! - Likelihood: x ~ N(θ, σ²_thermometer) with σ_thermometer = 0.5°C
//! - Prior: Conjugate prior with ν₀ = 1/1000 (very weak), μ₀ = 37.0°C
//! - Posterior: N(λₙ[0]/νₙ, σ²_thermometer/νₙ)

use dioxus::prelude::*;
use exponential::inference::ConjugatePrior;
use exponential::laws::gaussian::Normal1D;
use nalgebra::DVector;

/// Thermometer measurement with ID and value.
#[derive(Debug, Clone, PartialEq)]
struct Measurement {
    id: u64,
    value: f64,
}

/// Bayesian model for body temperature inference.
#[derive(Debug, Clone)]
struct BodyTempModel {
    /// Conjugate prior for the Normal distribution.
    prior: ConjugatePrior,
    /// Likelihood distribution (known variance).
    likelihood: Normal1D,
    /// All measurements taken.
    measurements: Vec<Measurement>,
    /// Next measurement ID.
    next_id: u64,
}

impl BodyTempModel {
    /// Create a new model with epidemiological prior.
    fn new() -> Self {
        // Thermometer uncertainty: σ = 0.5°C → σ² = 0.25
        let likelihood = Normal1D::new(0.0, 0.25).expect("valid variance");

        // Prior: ν₀ = 1/1000, μ₀ = 37.0°C
        let mu0 = 37.0;
        let nu0 = 0.001;
        // λ₀ = ν₀ * (μ₀, μ₀² + σ²)
        let lambda0 = DVector::from_vec(vec![nu0 * mu0, nu0 * (mu0 * mu0 + 0.25)]);
        let prior = ConjugatePrior::new(lambda0, nu0).expect("valid prior");

        Self {
            prior,
            likelihood,
            measurements: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a measurement and update posterior.
    fn add_measurement(&mut self, value: f64) {
        let id = self.next_id;
        self.next_id += 1;
        self.measurements.push(Measurement { id, value });
        self.update_posterior();
    }

    /// Remove a measurement by ID and update posterior.
    fn remove_measurement(&mut self, id: u64) {
        self.measurements.retain(|m| m.id != id);
        self.update_posterior();
    }

    /// Recompute posterior from all measurements.
    fn update_posterior(&mut self) {
        // Reset to prior
        let mu0 = 37.0;
        let nu0 = 0.001;
        let lambda0 = DVector::from_vec(vec![nu0 * mu0, nu0 * (mu0 * mu0 + 0.25)]);
        self.prior = ConjugatePrior::new(lambda0, nu0).expect("valid prior");

        // Update with all measurements
        let data: Vec<f64> = self.measurements.iter().map(|m| m.value).collect();
        if !data.is_empty() {
            self.prior = self
                .prior
                .update_batch(&self.likelihood, &data)
                .expect("valid update");
        }
    }

    /// Posterior mean (estimated true temperature).
    fn posterior_mean(&self) -> f64 {
        self.prior.posterior_mean()[0]
    }

    /// Posterior standard deviation.
    fn posterior_std(&self) -> f64 {
        let nu = self.prior.nu();
        (0.25 / nu).sqrt() // σ²_thermometer = 0.25
    }

    /// 95% credible interval: mean ± 1.96 * std.
    fn credible_interval(&self) -> (f64, f64) {
        let mean = self.posterior_mean();
        let std = self.posterior_std();
        (mean - 1.96 * std, mean + 1.96 * std)
    }

    /// Number of measurements.
    fn count(&self) -> usize {
        self.measurements.len()
    }
}

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let model = use_signal(BodyTempModel::new);

    rsx! {
        div {
            class: "min-h-screen bg-gray-50",
            Header {}
            main {
                class: "max-w-4xl mx-auto p-4",
                BodyTemperaturePage { model }
            }
        }
    }
}

#[component]
fn Header() -> Element {
    rsx! {
        header {
            class: "bg-white shadow",
            div {
                class: "max-w-4xl mx-auto px-4 py-6",
                h1 {
                    class: "text-3xl font-bold text-gray-900",
                    "🌡️ Body Temperature Bayesian Inference"
                }
                p {
                    class: "text-gray-600 mt-2",
                    "Estimate your true body temperature from thermometer measurements."
                }
            }
        }
    }
}

#[component]
fn BodyTemperaturePage(model: Signal<BodyTempModel>) -> Element {
    let mut new_value = use_signal(String::new);

    let on_add = move |_| {
        if let Ok(value) = new_value().parse::<f64>() {
            model.write().add_measurement(value);
            new_value.set(String::new());
        }
    };

    let on_remove = move |id: u64| {
        model.write().remove_measurement(id);
    };

    rsx! {
        div {
            class: "space-y-8",
            PosteriorDisplay { model }
            MeasurementForm {
                new_value,
                on_add,
            }
            MeasurementList {
                model,
                on_remove,
            }
            Explanation {}
        }
    }
}

#[component]
fn PosteriorDisplay(model: Signal<BodyTempModel>) -> Element {
    let mean = model().posterior_mean();
    let std = model().posterior_std();
    let (lower, upper) = model().credible_interval();
    let count = model().count();

    rsx! {
        div {
            class: "bg-white rounded-xl shadow p-6",
            h2 {
                class: "text-2xl font-bold text-gray-800 mb-4",
                "Posterior Distribution"
            }
            div {
                class: "grid grid-cols-1 md:grid-cols-3 gap-6",
                div {
                    class: "text-center p-4 bg-blue-50 rounded-lg",
                    p {
                        class: "text-sm text-blue-700 font-medium",
                        "Estimated Temperature"
                    }
                    p {
                        class: "text-3xl font-bold text-blue-900",
                        "{mean:.2} °C"
                    }
                }
                div {
                    class: "text-center p-4 bg-green-50 rounded-lg",
                    p {
                        class: "text-sm text-green-700 font-medium",
                        "Uncertainty (σ)"
                    }
                    p {
                        class: "text-3xl font-bold text-green-900",
                        "{std:.3} °C"
                    }
                }
                div {
                    class: "text-center p-4 bg-purple-50 rounded-lg",
                    p {
                        class: "text-sm text-purple-700 font-medium",
                        "95% Credible Interval"
                    }
                    p {
                        class: "text-3xl font-bold text-purple-900",
                        "[{lower:.2}, {upper:.2}] °C"
                    }
                }
            }
            p {
                class: "text-gray-600 mt-4 text-center",
                "Based on {count} measurement(s). Prior: N(37.0, 0.25) with ν₀=0.001."
            }
        }
    }
}

#[component]
fn MeasurementForm(new_value: Signal<String>, on_add: EventHandler<()>) -> Element {
    rsx! {
        div {
            class: "bg-white rounded-xl shadow p-6",
            h2 {
                class: "text-2xl font-bold text-gray-800 mb-4",
                "Add Measurement"
            }
            div {
                class: "flex gap-4",
                input {
                    r#type: "number",
                    step: "0.1",
                    placeholder: "Temperature in °C (e.g., 36.8)",
                    class: "flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500",
                    value: "{new_value}",
                    oninput: move |e| new_value.set(e.value()),
                    onkeydown: move |e| {
                        if e.key() == dioxus::prelude::Key::Enter {
                            on_add.call(());
                        }
                    },
                }
                button {
                    class: "px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500",
                    onclick: move |_| on_add.call(()),
                    "Add"
                }
            }
            p {
                class: "text-gray-500 text-sm mt-2",
                "Thermometer uncertainty: ±0.5°C. Enter a value and press Add or Enter."
            }
        }
    }
}

#[component]
fn MeasurementList(model: Signal<BodyTempModel>, on_remove: EventHandler<u64>) -> Element {
    let measurements = model().measurements.clone();

    rsx! {
        div {
            class: "bg-white rounded-xl shadow p-6",
            h2 {
                class: "text-2xl font-bold text-gray-800 mb-4",
                "Measurements ({measurements.len()})"
            }
            if measurements.is_empty() {
                p {
                    class: "text-gray-500 text-center py-8",
                    "No measurements yet. Add one above."
                }
            } else {
                ul {
                    class: "space-y-3",
                    for m in measurements {
                        li {
                            key: "{m.id}",
                            class: "flex items-center justify-between p-4 border border-gray-200 rounded-lg",
                            div {
                                class: "flex items-center gap-4",
                                span {
                                    class: "text-2xl text-gray-400",
                                    "🌡️"
                                }
                                div {
                                    span {
                                        class: "text-lg font-semibold text-gray-800",
                                        "{m.value:.2} °C"
                                    }
                                    p {
                                        class: "text-sm text-gray-500",
                                        "ID: {m.id}"
                                    }
                                }
                            }
                            button {
                                class: "px-4 py-2 text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg",
                                onclick: move |_| on_remove.call(m.id),
                                "Remove"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn Explanation() -> Element {
    rsx! {
        div {
            class: "bg-white rounded-xl shadow p-6",
            h2 {
                class: "text-2xl font-bold text-gray-800 mb-4",
                "How It Works"
            }
            div {
                class: "prose prose-blue max-w-none",
                p {
                    "This app uses Bayesian inference to estimate your true body temperature θ from thermometer measurements."
                }
                ul {
                    li {
                        strong { "Likelihood: " }
                        "Each measurement x is assumed to follow a Normal distribution N(θ, σ²) with σ = 0.5°C (thermometer uncertainty)."
                    }
                    li {
                        strong { "Prior: " }
                        "We start with a weak prior centered at 37.0°C (standard human body temperature) with pseudo‑observation count ν₀ = 1/1000."
                    }
                    li {
                        strong { "Posterior: " }
                        "After each measurement, we update the posterior distribution using the conjugate prior for the Normal distribution."
                    }
                    li {
                        strong { "Result: " }
                        "The posterior mean is our best estimate of θ; the posterior standard deviation quantifies our uncertainty."
                    }
                }
                p {
                    class: "text-sm text-gray-500 mt-4",
                    "The math is implemented in the `exponential` crate using the general exponential‑family conjugate‑prior framework."
                }
            }
        }
    }
}
