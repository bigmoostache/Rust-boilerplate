use dioxus::prelude::*;

use crate::chart::build_posterior_svg;
use crate::components::{MathInline, SectionWrap, retrigger_katex};
use crate::model::{ALPHA_0, MU_0, compute_posterior};

#[component]
pub fn Section7() -> Element {
    rsx! {
        SectionWrap { number: "§ 7", title: "Simulation interactive",
            p { "Ajoutez des mesures et observez la mise à jour bayésienne en temps réel. Les hyperparamètres ", MathInline { tex: r"\boldsymbol{{\lambda}}" }, " et ", MathInline { tex: r"\nu" }, " s\u{2019}accumulent ; le posterior (courbe rouge) se concentre autour de la vraie température." }
            BayesWidget {}
        }
    }
}

#[component]
fn BayesWidget() -> Element {
    let mut obs = use_signal(|| vec![38.2, 38.6, 38.1]);
    let mut new_val = use_signal(|| 38.2_f64);

    let data = obs();
    let post = compute_posterior(&data);
    let svg = build_posterior_svg(&post, &data);
    use_effect(retrigger_katex);

    let n = data.len();
    let stats: Vec<(&str, String, bool)> = if n > 0 {
        vec![
            ("n", format!("{n}"), false),
            ("\u{03bb}\u{2081}", format!("{:.2}", post.lambda1), false),
            ("\u{03bb}\u{2082}", format!("{:.1}", post.lambda2), false),
            ("\u{03bd}", format!("{:.2}", post.nu), false),
            ("\u{03bc} post", format!("{:.2}\u{00b0}C", post.mu), true),
            ("df = 2\u{03b1}", format!("{:.1}", 2.0 * post.alpha), false),
            ("\u{03c3} prior", "0.25\u{00b0}C".to_string(), false),
            (
                "\u{00f7}\u{03c3}",
                format!(
                    "\u{00d7}{:.0}",
                    0.25 / (post.beta / (post.alpha * post.nu)).sqrt()
                ),
                false,
            ),
        ]
    } else {
        vec![
            ("n", "0".to_string(), false),
            ("\u{03bb}\u{2081}", "\u{2014}".to_string(), false),
            ("\u{03bb}\u{2082}", "\u{2014}".to_string(), false),
            ("\u{03bd}", "\u{2014}".to_string(), false),
            ("\u{03bc} post", format!("{MU_0}\u{00b0}C"), true),
            ("df = 2\u{03b1}", format!("{:.1}", 2.0 * ALPHA_0), false),
            ("\u{03c3} prior", "0.25\u{00b0}C".to_string(), false),
            ("\u{00f7}\u{03c3}", "\u{2014}".to_string(), false),
        ]
    };

    rsx! {
        div { class: "widget",
            div { class: "widget-title", "Simulation interactive — mise à jour bayésienne" }

            // Controls
            div { class: "widget-controls",
                div { class: "control-group",
                    label { "Nouvelle observation" }
                    input {
                        r#type: "range", min: "35", max: "41", step: "0.1",
                        value: "{new_val}",
                        oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { new_val.set(v); } },
                    }
                    span { class: "val", "{new_val:.1}°C" }
                }
                button { class: "add-obs-btn",
                    onclick: move |_| { let v = new_val(); obs.write().push(v); },
                    "+ Ajouter"
                }
                button { class: "reset-btn",
                    onclick: move |_| { *obs.write() = vec![38.2, 38.6, 38.1]; },
                    "Réinitialiser"
                }
            }

            // Stats grid
            div { class: "stats-grid",
                for (label, value, hi) in stats {
                    div { class: if hi { "stat-card highlight" } else { "stat-card" },
                        div { class: "stat-label", "{label}" }
                        div { class: "stat-value", "{value}" }
                    }
                }
            }

            // Observation tags
            div { class: "obs-list",
                span { class: "obs-label", "observations (clic pour supprimer) :" }
                for (i, x) in data.iter().enumerate() {
                    span { class: "obs-tag",
                        onclick: {
                            let idx = i;
                            move |_| { obs.write().remove(idx); }
                        },
                        "{x:.1}°C ×"
                    }
                }
            }

            // SVG Chart
            div { class: "posterior-svg-wrap",
                div { dangerous_inner_html: "{svg}" }
            }
            div { class: "plot-legend",
                "— — prior (large)   —— Student t(2α) exact   - - Gaussienne approx   | 38°C   ↓ obs"
            }
        }
    }
}
