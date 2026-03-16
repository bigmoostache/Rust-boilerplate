use dioxus::prelude::*;

use crate::components::{Boxed, Callout, MathBlock, MathInline, SectionWrap};

// ─── §1 ─────────────────────────────────────────────────────────

#[component]
pub fn Section1() -> Element {
    rsx! {
        SectionWrap { number: "§ 1", title: "La famille exponentielle",
            p {
                "Une distribution appartient à la ", em { "famille exponentielle" },
                " si sa densité s\u{2019}écrit sous la forme canonique :"
            }
            MathBlock { label: "Définition — forme canonique",
                tex: r"p(x \mid \boldsymbol{{\eta}}) \;=\; h(x)\,\exp\!\bigl(\boldsymbol{{\eta}}^\top T(x) - A(\boldsymbol{{\eta}})\bigr)"
            }
            p {
                "Les trois ingrédients : ", MathInline { tex: r"h(x)" },
                " la mesure de base, ", MathInline { tex: r"T(x)" },
                " la ", em { "statistique suffisante" }, ", et ",
                MathInline { tex: r"\boldsymbol{{\eta}}" }, " le ",
                em { "paramètre naturel" }, ". La ", em { "log-partition" },
                " ", MathInline { tex: r"A(\boldsymbol{{\eta}})" },
                " assure la normalisation :"
            }
            MathBlock { tex: r"A(\boldsymbol{{\eta}}) = \log \int h(x)\,\exp\!\bigl(\boldsymbol{{\eta}}^\top T(x)\bigr)\,dx" }
            p { "Ses dérivées donnent les moments de ", MathInline { tex: r"T(x)" }, " :" }
            MathBlock { tex: r"\nabla A(\boldsymbol{{\eta}}) = \mathbb{{E}}[T(x)], \qquad \nabla^2 A(\boldsymbol{{\eta}}) = \mathrm{{Var}}[T(x)]" }
            Callout {
                strong { "Intuition clé. " }
                "Toute l\u{2019}information sur ", MathInline { tex: r"\boldsymbol{{\eta}}" },
                " contenue dans une observation ", MathInline { tex: r"x" },
                " est capturée par ", MathInline { tex: r"T(x)" },
                ". Deux observations avec le même ", MathInline { tex: r"T(x)" },
                " sont ", em { "équivalentes" }, " pour l\u{2019}inférence."
            }
        }
    }
}

// ─── §2 ─────────────────────────────────────────────────────────

#[component]
pub fn Section2() -> Element {
    rsx! {
        SectionWrap { number: "§ 2", title: "Vraisemblance de n observations",
            p {
                "Pour ", MathInline { tex: r"n" },
                " observations i.i.d., la vraisemblance se factorise élégamment :"
            }
            MathBlock {
                tex: r"p(x_1,\ldots,x_n \mid \boldsymbol{{\eta}}) = \prod_i h(x_i) \;\cdot\; \exp\!\Bigl(\boldsymbol{{\eta}}^\top \sum_i T(x_i) - n\,A(\boldsymbol{{\eta}})\Bigr)"
            }
            p {
                "La vraisemblance ne dépend des données qu\u{2019}à travers ",
                MathInline { tex: r"\textstyle\sum_i T(x_i)" },
                ". C\u{2019}est le théorème de suffisance de Fisher — ce vecteur de dimension fixe résume intégralement l\u{2019}information de ",
                MathInline { tex: r"n" }, " observations, quelle que soit ",
                MathInline { tex: r"n" }, "."
            }
        }
    }
}

// ─── §3 ─────────────────────────────────────────────────────────

#[component]
pub fn Section3() -> Element {
    rsx! {
        SectionWrap { number: "§ 3", title: "Construction du prior conjugué joint",
            p {
                "On cherche un prior ", MathInline { tex: r"p(\boldsymbol{{\eta}})" },
                " tel que le posterior reste dans la ",
                em { "même famille paramétrique" }, ". La forme conjuguée naturelle est :"
            }
            Boxed { title: "Théorème — Prior conjugué universel",
                MathBlock {
                    tex: r"p(\boldsymbol{{\eta}} \mid \boldsymbol{{\lambda}}_0,\,\nu_0) \;\propto\; \exp\!\bigl(\boldsymbol{{\eta}}^\top \boldsymbol{{\lambda}}_0 - \nu_0\,A(\boldsymbol{{\eta}})\bigr), \quad \boldsymbol{{\lambda}}_0 \in \mathbb{{R}}^k,\; \nu_0 > 0"
                }
            }
            p {
                "Ce prior est normalisable si et seulement si ",
                MathInline { tex: r"\nu_0 > 0" }, " et ",
                MathInline { tex: r"\boldsymbol{{\lambda}}_0/\nu_0" },
                " est dans l\u{2019}intérieur du domaine naturel de ",
                MathInline { tex: r"\boldsymbol{{\eta}}" }, "."
            }
            h3 { "Interprétation des hyperparamètres" }
            DataTable3 {}
            Callout {
                strong { "λ comme mémoire suffisante. " }
                MathInline { tex: r"\boldsymbol{{\lambda}}" },
                " est une ", em { "urne" },
                " qui accumule les statistiques suffisantes. ",
                MathInline { tex: r"\nu" }, " est son poids. ",
                MathInline { tex: r"\boldsymbol{{\lambda}}/\nu" },
                " est sa moyenne. Le prior initialise l\u{2019}urne avec ",
                MathInline { tex: r"\nu_0" }, " billes fictives ; chaque observation verse ",
                MathInline { tex: r"T(x_i)" },
                " dans l\u{2019}urne et incrémente ", MathInline { tex: r"\nu" }, " d\u{2019}une unité."
            }
        }
    }
}

#[component]
fn DataTable3() -> Element {
    rsx! {
        table { class: "data-table",
            thead { tr {
                th { "Hyperparamètre" } th { "Interprétation" } th { "Analogie" }
            }}
            tbody {
                tr { td { MathInline { tex: r"\boldsymbol{{\lambda}}_0" } } td { "Statistiques suffisantes fictives a priori" } td { "Billes initiales dans une urne" } }
                tr { td { MathInline { tex: r"\nu_0" } } td { "Poids du prior (observations fictives)" } td { "Nombre de billes initiales" } }
                tr { td { MathInline { tex: r"\boldsymbol{{\lambda}}_0 / \nu_0" } } td { "Valeur centrale a priori de η" } td { "Moyenne des billes initiales" } }
            }
        }
    }
}

// ─── §4 ─────────────────────────────────────────────────────────

#[component]
pub fn Section4() -> Element {
    rsx! {
        SectionWrap { number: "§ 4", title: "La mise à jour bayésienne — deux additions",
            p { "Le posterior s\u{2019}obtient par le théorème de Bayes :" }
            MathBlock {
                tex: r"p(\boldsymbol{{\eta}} \mid \mathbf{{x}}) \;\propto\; \exp\!\Bigl(\boldsymbol{{\eta}}^\top \bigl(\boldsymbol{{\lambda}}_0 + \textstyle\sum_i T(x_i)\bigr) - (\nu_0 + n)\,A(\boldsymbol{{\eta}})\Bigr)"
            }
            Boxed { title: "Règle de mise à jour — universelle",
                MathBlock {
                    tex: r"\boldsymbol{{\lambda}}_n = \boldsymbol{{\lambda}}_0 + \sum_i T(x_i), \qquad \nu_n = \nu_0 + n"
                }
            }
            p { "C\u{2019}est la beauté de la conjugaison : toute la complexité bayésienne se réduit à deux additions." }
            h3 { "Structure de shrinkage" }
            p {
                "Le ratio ", MathInline { tex: r"\boldsymbol{{\lambda}}_n / \nu_n" },
                " interpole entre prior et données :"
            }
            MathBlock {
                tex: r"\frac{{\boldsymbol{{\lambda}}_n}}{{\nu_n}} = \frac{{\nu_0}}{{\nu_0+n}} \cdot \frac{{\boldsymbol{{\lambda}}_0}}{{\nu_0}} + \frac{{n}}{{\nu_0+n}} \cdot \bar{{T}}"
            }
            p {
                "Quand ", MathInline { tex: r"n \to \infty" },
                ", les données écrasent le prior (",
                MathInline { tex: r"\boldsymbol{{\lambda}}_n/\nu_n \to \bar{{T}}" },
                ") et on retrouve l\u{2019}estimateur fréquentiste."
            }
        }
    }
}

// ─── §5 ─────────────────────────────────────────────────────────

#[component]
pub fn Section5() -> Element {
    rsx! {
        SectionWrap { number: "§ 5", title: "Application : la loi gaussienne",
            p {
                "On pose ", MathInline { tex: r"X \sim \mathcal{{N}}(\mu,\,\sigma^2)" },
                ". La forme canonique donne :"
            }
            MathBlock { label: "Identification des composantes",
                tex: r"h(x) = \frac{{1}}{{\sqrt{{2\pi}}}}, \quad T(x) = \begin{{pmatrix}} x \\ x^2 \end{{pmatrix}}, \quad \boldsymbol{{\eta}} = \begin{{pmatrix}} \mu/\sigma^2 \\ -1/(2\sigma^2) \end{{pmatrix}}"
            }
            MathBlock {
                tex: r"A(\boldsymbol{{\eta}}) = -\frac{{\eta_1^2}}{{4\,\eta_2}} + \tfrac{{1}}{{2}}\ln\!\Bigl(-\frac{{\pi}}{{\eta_2}}\Bigr)"
            }
            p { "La bijection s\u{2019}inverse :" }
            MathBlock {
                tex: r"\sigma^2 = -\frac{{1}}{{2\,\eta_2}}, \qquad \mu = -\frac{{\eta_1}}{{2\,\eta_2}}"
            }
            p {
                "En revenant aux paramètres usuels, le prior conjugué se réduit à la loi ",
                strong { "Normale-Gamma Inverse" },
                " ", MathInline { tex: r"\mathrm{{NIG}}(\mu_0,\,\nu_0,\,\alpha_0,\,\beta_0)" }, "."
            }
            h3 { "Statistiques suffisantes gaussiennes" }
            DataTable5 {}
        }
    }
}

#[component]
fn DataTable5() -> Element {
    rsx! {
        table { class: "data-table",
            thead { tr { th { "Composante" } th { "Statistique" } th { "Information portée" } }}
            tbody {
                tr { td { MathInline { tex: r"T^{{(1)}}(x) = x" } } td { "valeur brute" } td { "Localisation → μ" } }
                tr { td { MathInline { tex: r"T^{{(2)}}(x) = x^2" } } td { "carré" } td { "Dispersion → σ²" } }
                tr { td { MathInline { tex: r"\sum x_i,\;\sum x_i^2" } } td { "sommes" } td { "Tout ce qu\u{2019}il faut pour inférer (μ, σ²)" } }
            }
        }
    }
}

// ─── §6 ─────────────────────────────────────────────────────────

#[component]
pub fn Section6() -> Element {
    rsx! {
        SectionWrap { number: "§ 6", title: "Cas pratique : thermométrie bayésienne",
            p {
                "On mesure la température frontale d\u{2019}un patient avec un thermomètre d\u{2019}incertitude ±0.5°C, et un prior épidémiologique très faible (",
                MathInline { tex: r"\nu_0 = 0.001" }, ")."
            }
            h3 { "Calibration du prior" }
            p {
                "L\u{2019}incertitude ±0.5°C correspond à environ ", MathInline { tex: r"2\sigma" },
                ", donc ", MathInline { tex: r"\sigma \approx 0.25\,°\mathrm{{C}}" },
                " ⇒ ", MathInline { tex: r"\sigma^2 \approx 0.0625" },
                ". Avec ", MathInline { tex: r"\mu_0 = 36.2\,°\mathrm{{C}}" }, " :"
            }
            MathBlock {
                tex: r"\boldsymbol{{\lambda}}_0 = \nu_0 \begin{{pmatrix}} \mu_0/\sigma_0^2 \\ -1/(2\sigma_0^2) \end{{pmatrix}} = \begin{{pmatrix}} 0.579 \\ -0.008 \end{{pmatrix}}, \quad \nu_0 = 0.001"
            }
            h3 { "Trois mesures, deux additions" }
            MeasurementTable6 {}
            Boxed { title: "Mise à jour",
                MathBlock {
                    tex: r"\boldsymbol{{\lambda}}_3 = \begin{{pmatrix}} 0.579 + 114.9 \\ -0.008 - 4400.85 \end{{pmatrix}} = \begin{{pmatrix}} 115.479 \\ -4400.858 \end{{pmatrix}}, \quad \nu_3 = 3.001"
                }
            }
            h3 { "Retour vers (μ, σ²)" }
            MathBlock {
                tex: r"\sigma^2_{{post}} = -\frac{{1}}{{2\,\eta_2}} \approx 3.41 \times 10^{{-4}} \;\Rightarrow\; \sigma_{{post}} \approx 0.018\,°\mathrm{{C}}, \qquad \mu_{{post}} \approx 38.30\,°\mathrm{{C}}"
            }
            Callout {
                strong { "Réduction d\u{2019}incertitude. " }
                "Trois mesures ont réduit ", MathInline { tex: r"\sigma" },
                " de 0.25°C à 0.018°C — un facteur ×14. Le patient est clairement fébrile avec une certitude quasi-totale."
            }
        }
    }
}

#[component]
fn MeasurementTable6() -> Element {
    rsx! {
        table { class: "data-table",
            thead { tr { th { "Mesure" } th { MathInline { tex: r"x_i\;(°\mathrm{{C}})" } } th { MathInline { tex: r"x_i^2" } } }}
            tbody {
                tr { td { "1" } td { "38.2" } td { "1459.24" } }
                tr { td { "2" } td { "38.6" } td { "1490.00" } }
                tr { td { "3" } td { "38.1" } td { "1451.61" } }
                tr { td { strong { "Σ" } } td { class: "accent", strong { "114.9" } } td { class: "accent", strong { "4400.85" } } }
            }
        }
    }
}

// ─── §8 ─────────────────────────────────────────────────────────

#[component]
pub fn Section8() -> Element {
    rsx! {
        SectionWrap { number: "§ 8", title: "Synthèse — ce que la structure révèle",
            p {
                "La conjugaison dans les familles exponentielles n\u{2019}est pas une coïncidence algébrique : c\u{2019}est une conséquence directe du théorème de suffisance. La vraisemblance ne dépend des données qu\u{2019}à travers ",
                MathInline { tex: r"T(x)" },
                " ; le prior conjugué a ", em { "la même dépendance fonctionnelle" },
                " en ", MathInline { tex: r"\boldsymbol{{\eta}}" },
                " via ", MathInline { tex: r"A(\boldsymbol{{\eta}})" }, "."
            }
            DataTable8 {}
            Callout {
                strong { "Le résultat le plus profond. " }
                "Dans la limite ", MathInline { tex: r"\nu_0 \to 0" },
                " (prior non-informatif), les estimateurs bayésiens convergent vers les estimateurs du maximum de vraisemblance — le fréquentisme émerge comme cas limite du bayésien."
            }
        }
    }
}

#[component]
fn DataTable8() -> Element {
    rsx! {
        table { class: "data-table",
            thead { tr { th { "Objet" } th { "Rôle" } th { "Intuition" } }}
            tbody {
                tr { td { MathInline { tex: r"T(x)" } } td { "Statistique suffisante" } td { "Ce que x apprend sur η" } }
                tr { td { MathInline { tex: r"A(\boldsymbol{{\eta}})" } } td { "Log-partition" } td { "Géométrie de la famille" } }
                tr { td { MathInline { tex: r"\boldsymbol{{\lambda}}" } } td { "Mémoire accumulée" } td { "Urne de statistiques suffisantes" } }
                tr { td { MathInline { tex: r"\nu" } } td { "Poids de la mémoire" } td { "Compteur d\u{2019}observations" } }
                tr { td { MathInline { tex: r"\boldsymbol{{\lambda}}/\nu" } } td { "Estimateur courant" } td { "Moyenne de l\u{2019}urne (shrinkage)" } }
                tr { td { MathInline { tex: r"\boldsymbol{{\lambda}}_n = \boldsymbol{{\lambda}}_0 + \sum T(x_i)" } } td { "Mise à jour" } td { "Deux additions" } }
            }
        }
    }
}

// ─── Header / Footer ────────────────────────────────────────────

#[component]
pub fn ChapterHeader() -> Element {
    rsx! {
        div { class: "chapter-header",
            div { class: "chapter-label", "Inférence Bayésienne · Chapitre 4" }
            h1 { class: "chapter-title",
                "Le Prior Conjugué"
                br {}
                "dans les Familles Exponentielles"
            }
            p { class: "chapter-subtitle",
                "De la structure abstraite d\u{2019}une famille exponentielle jusqu\u{2019}à l\u{2019}estimation bayésienne de la température d\u{2019}un patient fébrile — un fil conducteur unique."
            }
        }
    }
}

#[component]
pub fn ChapterFooter() -> Element {
    rsx! {
        div { class: "chapter-footer",
            span { class: "footer-label", "Inférence Bayésienne · Chapitre 4" }
            span { class: "footer-label", "Prior Conjugué · Familles Exponentielles" }
        }
    }
}
