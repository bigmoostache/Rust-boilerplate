use dioxus::prelude::*;

#[component]
pub fn SectionWrap(number: String, title: String, children: Element) -> Element {
    rsx! {
        div { class: "section visible",
            div { class: "section-number", "{number}" }
            h2 { "{title}" }
            {children}
        }
    }
}

#[component]
pub fn MathBlock(#[props(default)] label: String, children: Element) -> Element {
    rsx! {
        div { class: "math-block",
            if !label.is_empty() {
                div { class: "math-label", "{label}" }
            }
            div { class: "math-display", {children} }
        }
    }
}

#[component]
pub fn MathInline(children: Element) -> Element {
    rsx! { code { class: "math-inline", {children} } }
}

#[component]
pub fn Boxed(title: String, children: Element) -> Element {
    rsx! {
        div { class: "boxed",
            div { class: "box-title", "{title}" }
            div { class: "math-display", {children} }
        }
    }
}

#[component]
pub fn Callout(children: Element) -> Element {
    rsx! { div { class: "callout", {children} } }
}

#[component]
pub fn Hr() -> Element {
    rsx! { hr { class: "rule" } }
}
