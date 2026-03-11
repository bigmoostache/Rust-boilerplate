#!/usr/bin/env bash
# Project rules callback — structure & protected files checks.
# Input: $CP_CHANGED_FILES (newline-separated list of changed files)
# Called from project root.
set -euo pipefail

# ── 1. File lengths ──────────────────────────────────────────────────
bash scripts/check-file-lengths.sh 2>&1

# ── 2. Folder sizes ─────────────────────────────────────────────────
bash scripts/check-folder-sizes.sh 2>&1

# ── 3. Lint exceptions ──────────────────────────────────────────────
bash scripts/check-lint-exceptions.sh 2>&1

# ── 4. Protected files chain ────────────────────────────────────────
if ! bash scripts/check-lint-config.sh 2>&1; then
  # Build the list of protected files dynamically from the YAML manifest.
  protected_list=$(grep -E '^\s*-?\s*path:' scripts/protected-files.yaml \
    | sed 's/.*path:\s*/  - /' \
    | sed 's/^  - \s*/  - /')

  cat <<EOF

╔══════════════════════════════════════════════════════════════════════╗
║                   PROTECTED FILE MODIFICATION DETECTED              ║
╚══════════════════════════════════════════════════════════════════════╝

You have edited a protected file. Reminder: protected files are
${protected_list}

This guardrail is here for multiple reasons:
  - Protecting the hard-won ultra-severe set of linters from your
    natural tendency to relax those rules to code more easily.
  - Keep the list of exceptions to those rules as limited as possible.
  - Preventing you from cheating the above by editing scripts that
    enforce those rules.

If your edit is legit, please ask the human to review your changes
and sign the new chain of documents with the password you do not
have access to.
EOF
  exit 1
fi
