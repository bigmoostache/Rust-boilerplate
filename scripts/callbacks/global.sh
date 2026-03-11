#!/usr/bin/env bash
# Global callback — runs once per edit batch.
# Input: $CP_CHANGED_FILES (newline-separated list of changed files)
# Called from project root.
set -euo pipefail

# ── 1. Cargo check ────────────────────────────────────────────────────
cargo check --workspace --quiet 2>&1

# ── 2. Clippy ─────────────────────────────────────────────────────────
cargo clippy --workspace --quiet -- -D warnings 2>&1

# ── 3. Tests ──────────────────────────────────────────────────────────
cargo test --workspace --quiet 2>&1
