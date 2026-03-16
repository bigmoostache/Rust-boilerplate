use dioxus::prelude::*;

mod chart;
mod components;
mod math;
mod model;
mod sections;
mod widget;

use components::Hr;
use sections::{
    ChapterFooter, ChapterHeader, Section1, Section2, Section3, Section4, Section5, Section6,
    Section8,
};
use widget::Section7;

const STYLE: Asset = asset!("/assets/style.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    rsx! {
        document::Stylesheet { href: STYLE }
        div { class: "page",
            ChapterHeader {}
            Section1 {}
            Hr {}
            Section2 {}
            Hr {}
            Section3 {}
            Hr {}
            Section4 {}
            Hr {}
            Section5 {}
            Hr {}
            Section6 {}
            Hr {}
            Section7 {}
            Hr {}
            Section8 {}
            ChapterFooter {}
        }
    }
}
