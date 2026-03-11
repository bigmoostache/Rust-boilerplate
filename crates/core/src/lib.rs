//! Core library for the Baillissime application.

/// Returns a greeting for the given name.
///
/// # Examples
///
/// ```
/// let g = app_core::greet("Alice");
/// assert_eq!(g, "Hello, Alice!");
/// ```
#[must_use]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet("World"), "Hello, World!");
    }
}
