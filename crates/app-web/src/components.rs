use dioxus::prelude::*;

/// Trigger KaTeX auto-render on the page. Call after any content change.
pub fn retrigger_katex() {
    document::eval(
        r#"
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }
        "#,
    );
}

#[component]
pub fn SectionWrap(number: String, title: String, children: Element) -> Element {
    // Re-trigger KaTeX whenever a section renders
    use_effect(retrigger_katex);
    rsx! {
        div { class: "section visible",
            div { class: "section-number", "{number}" }
            h2 { "{title}" }
            {children}
        }
    }
}

#[component]
pub fn MathBlock(#[props(default)] label: String, tex: String) -> Element {
    let html = format!("$${tex}$$");
    rsx! {
        div { class: "math-block",
            if !label.is_empty() {
                div { class: "math-label", "{label}" }
            }
            div { class: "math-display", dangerous_inner_html: "{html}" }
        }
    }
}

#[component]
pub fn MathInline(tex: String) -> Element {
    let html = format!("${tex}$");
    rsx! { span { class: "math-inline", dangerous_inner_html: "{html}" } }
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
