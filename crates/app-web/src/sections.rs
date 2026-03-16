use dioxus::prelude::*;

use crate::components::{Boxed, Callout, MathBlock, MathInline, SectionWrap};

// ─── §1 ─────────────────────────────────────────────────────────

#[component]
pub fn Section1() -> Element {
    rsx! {
        SectionWrap { number: "§ 1", title: "La famille exponentielle",
            p { "Une distribution appartient à la ", em { "famille exponentielle" }, " si sa densité s\u{2019}écrit sous la forme canonique :" }
            MathBlock { label: "Définition — forme canonique",
                "p(x | η) = h(x) · exp( ηᵀ T(x) − A(η) )"
            }
            p { "Les trois ingrédients : ", MathInline { "h(x)" }, " la mesure de base, ", MathInline { "T(x)" }, " la ", em { "statistique suffisante" }, ", et ", MathInline { "η" }, " le ", em { "paramètre naturel" }, ". La ", em { "log-partition" }, " ", MathInline { "A(η)" }, " assure la normalisation :" }
            MathBlock { "A(η) = log ∫ h(x) · exp( ηᵀ T(x) ) dx" }
            p { "Ses dérivées donnent les moments de ", MathInline { "T(x)" }, " :" }
            MathBlock { "∇ A(η) = E[T(x)],    ∇² A(η) = Var[T(x)]" }
            Callout {
                strong { "Intuition clé. " }
                "Toute l\u{2019}information sur ", MathInline { "η" }, " contenue dans une observation ", MathInline { "x" }, " est capturée par ", MathInline { "T(x)" }, ". Deux observations avec le même ", MathInline { "T(x)" }, " sont ", em { "équivalentes" }, " pour l\u{2019}inférence."
            }
        }
    }
}

// ─── §2 ─────────────────────────────────────────────────────────

#[component]
pub fn Section2() -> Element {
    rsx! {
        SectionWrap { number: "§ 2", title: "Vraisemblance de n observations",
            p { "Pour ", MathInline { "n" }, " observations i.i.d., la vraisemblance se factorise élégamment :" }
            MathBlock { "p(x₁,…,xₙ | η) = ∏ᵢ h(xᵢ) · exp( ηᵀ Σᵢ T(xᵢ) − n A(η) )" }
            p { "La vraisemblance ne dépend des données qu\u{2019}à travers ", MathInline { "Σᵢ T(xᵢ)" }, ". C\u{2019}est le théorème de suffisance de Fisher — ce vecteur de dimension fixe résume intégralement l\u{2019}information de ", MathInline { "n" }, " observations, quelle que soit ", MathInline { "n" }, "." }
        }
    }
}

// ─── §3 ─────────────────────────────────────────────────────────

#[component]
pub fn Section3() -> Element {
    rsx! {
        SectionWrap { number: "§ 3", title: "Construction du prior conjugué joint",
            p { "On cherche un prior ", MathInline { "p(η)" }, " tel que le posterior reste dans la ", em { "même famille paramétrique" }, ". La forme conjuguée naturelle est :" }
            Boxed { title: "Théorème — Prior conjugué universel",
                "p(η | λ₀, ν₀) ∝ exp( ηᵀ λ₀ − ν₀ A(η) ),    λ₀ ∈ ℝᵏ, ν₀ > 0"
            }
            p { "Ce prior est normalisable si et seulement si ", MathInline { "ν₀ > 0" }, " et ", MathInline { "λ₀/ν₀" }, " est dans l\u{2019}intérieur du domaine naturel de ", MathInline { "η" }, "." }
            h3 { "Interprétation des hyperparamètres" }
            DataTable3 {}
            Callout {
                strong { "λ comme mémoire suffisante. " }
                MathInline { "λ" }, " est une ", em { "urne" }, " qui accumule les statistiques suffisantes. ", MathInline { "ν" }, " est son poids. ", MathInline { "λ/ν" }, " est sa moyenne. Le prior initialise l\u{2019}urne avec ", MathInline { "ν₀" }, " billes fictives ; chaque observation verse ", MathInline { "T(xᵢ)" }, " dans l\u{2019}urne et incrémente ", MathInline { "ν" }, " d\u{2019}une unité."
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
                tr { td { MathInline { "λ₀" } } td { "Statistiques suffisantes fictives a priori" } td { "Billes initiales dans une urne" } }
                tr { td { MathInline { "ν₀" } } td { "Poids du prior (observations fictives)" } td { "Nombre de billes initiales" } }
                tr { td { MathInline { "λ₀/ν₀" } } td { "Valeur centrale a priori de η" } td { "Moyenne des billes initiales" } }
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
            MathBlock { "p(η | x) ∝ exp( ηᵀ Σᵢ T(xᵢ) − n A(η) ) · exp( ηᵀ λ₀ − ν₀ A(η) )\n       = exp( ηᵀ (λ₀ + Σᵢ T(xᵢ)) − (ν₀ + n) A(η) )" }
            Boxed { title: "Règle de mise à jour — universelle",
                "λₙ = λ₀ + Σᵢ T(xᵢ),    νₙ = ν₀ + n"
            }
            p { "C\u{2019}est la beauté de la conjugaison : toute la complexité bayésienne se réduit à deux additions." }
            h3 { "Structure de shrinkage" }
            p { "Le ratio ", MathInline { "λₙ/νₙ" }, " interpole entre prior et données :" }
            MathBlock { "λₙ/νₙ = (ν₀/(ν₀+n)) · (λ₀/ν₀) + (n/(ν₀+n)) · T̄" }
            p { "Quand ", MathInline { "n → ∞" }, ", les données écrasent le prior (", MathInline { "λₙ/νₙ → T̄" }, ") et on retrouve l\u{2019}estimateur fréquentiste." }
        }
    }
}

// ─── §5 ─────────────────────────────────────────────────────────

#[component]
pub fn Section5() -> Element {
    rsx! {
        SectionWrap { number: "§ 5", title: "Application : la loi gaussienne",
            p { "On pose ", MathInline { "X ~ N(μ, σ²)" }, ". La forme canonique donne :" }
            MathBlock { label: "Identification des composantes",
                "h(x) = 1/√(2π),  T(x) = (x, x²)ᵀ\nη = (μ/σ², −1/(2σ²))ᵀ"
            }
            MathBlock { "A(η) = −η₁²/(4η₂) + ½ log(−π/η₂)" }
            p { "La bijection s\u{2019}inverse :" }
            MathBlock { "σ² = −1/(2η₂),    μ = −η₁/(2η₂)" }
            p { "En revenant aux paramètres usuels, le prior conjugué se réduit à la loi ", strong { "Normale-Gamma Inverse" }, " NIG(μ₀, ν₀, α₀, β₀)." }
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
                tr { td { MathInline { "T⁽¹⁾(x) = x" } } td { "valeur brute" } td { "Localisation → μ" } }
                tr { td { MathInline { "T⁽²⁾(x) = x²" } } td { "carré" } td { "Dispersion → σ²" } }
                tr { td { MathInline { "Σxᵢ, Σxᵢ²" } } td { "sommes" } td { "Tout ce qu\u{2019}il faut pour inférer (μ, σ²)" } }
            }
        }
    }
}

// ─── §6 ─────────────────────────────────────────────────────────

#[component]
pub fn Section6() -> Element {
    rsx! {
        SectionWrap { number: "§ 6", title: "Cas pratique : thermométrie bayésienne",
            p { "On mesure la température frontale d\u{2019}un patient avec un thermomètre d\u{2019}incertitude ±0.5°C, et un prior épidémiologique très faible (", MathInline { "ν₀ = 0.001" }, ")." }
            h3 { "Calibration du prior" }
            p { "L\u{2019}incertitude ±0.5°C correspond à environ 2σ, donc σ ≈ 0.25°C ⇒ σ² ≈ 0.0625. Avec μ₀ = 36.2°C :" }
            MathBlock {
                "λ₀ = ν₀ · (μ₀/σ₀², −1/(2σ₀²))ᵀ = (0.579, −0.008)ᵀ\nν₀ = 0.001"
            }
            h3 { "Trois mesures, deux additions" }
            MeasurementTable6 {}
            Boxed { title: "Mise à jour",
                "λ₃ = (0.579 + 114.9, −0.008 − 4400.85)ᵀ = (115.479, −4400.858)ᵀ\nν₃ = 3.001"
            }
            h3 { "Retour vers (μ, σ²)" }
            MathBlock {
                "η₁ᵖᵒˢᵗ = 115.479/3.001 = 38.480\nη₂ᵖᵒˢᵗ = −4400.858/3.001 = −1466.46\nσ²ₚₒₛₜ = −1/(2η₂) ≈ 3.41×10⁻⁴  ⇒  σₚₒₛₜ ≈ 0.018°C\nμₚₒₛₜ = η₁ · σ² ≈ 38.30°C"
            }
            Callout {
                strong { "Réduction d\u{2019}incertitude. " }
                "Trois mesures ont réduit σ de 0.25°C à 0.018°C — un facteur ×14. Le patient est clairement fébrile avec une certitude quasi-totale."
            }
        }
    }
}

#[component]
fn MeasurementTable6() -> Element {
    rsx! {
        table { class: "data-table",
            thead { tr { th { "Mesure" } th { "xᵢ (°C)" } th { "xᵢ²" } }}
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
            p { "La conjugaison dans les familles exponentielles n\u{2019}est pas une coïncidence algébrique : c\u{2019}est une conséquence directe du théorème de suffisance. La vraisemblance ne dépend des données qu\u{2019}à travers ", MathInline { "T(x)" }, " ; le prior conjugué a ", em { "la même dépendance fonctionnelle" }, " en ", MathInline { "η" }, " via ", MathInline { "A(η)" }, "." }
            DataTable8 {}
            Callout {
                strong { "Le résultat le plus profond. " }
                "Dans la limite ", MathInline { "ν₀ → 0" }, " (prior non-informatif), les estimateurs bayésiens convergent vers les estimateurs du maximum de vraisemblance — le fréquentisme émerge comme cas limite du bayésien. La structure des familles exponentielles rend cette convergence transparente."
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
                tr { td { MathInline { "T(x)" } } td { "Statistique suffisante" } td { "Ce que x apprend sur η" } }
                tr { td { MathInline { "A(η)" } } td { "Log-partition" } td { "Géométrie de la famille" } }
                tr { td { MathInline { "λ" } } td { "Mémoire accumulée" } td { "Urne de statistiques suffisantes" } }
                tr { td { MathInline { "ν" } } td { "Poids de la mémoire" } td { "Compteur d\u{2019}observations" } }
                tr { td { MathInline { "λ/ν" } } td { "Estimateur courant" } td { "Moyenne de l\u{2019}urne (shrinkage)" } }
                tr { td { MathInline { "λₙ = λ₀ + Σ T(xᵢ)" } } td { "Mise à jour bayésienne" } td { "Deux additions, c\u{2019}est tout" } }
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
