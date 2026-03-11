#!/usr/bin/env bash
# Local callback — runs once per changed file.
# Input: $CP_CHANGED_FILE (single file path)
# Called from project root.
set -euo pipefail

file="$CP_CHANGED_FILE"

# Only act on Rust source files.
[[ "$file" == *.rs ]] || exit 0

# ── 1. Format the file ───────────────────────────────────────────────
rustfmt --edition 2024 "$file" 2>&1

# ── 2. Check line count ──────────────────────────────────────────────
lines=$(wc -l < "$file")
if [ "$lines" -gt 500 ]; then
  echo "FAIL: $file has $lines lines (max 500)" >&2
  echo "  → Refactor: extract into a sibling module." >&2
  exit 1
fi
