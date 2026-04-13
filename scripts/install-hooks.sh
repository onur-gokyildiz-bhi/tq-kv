#!/bin/bash
# Point this clone's git hooks at .githooks/ (tracked in the repo).
#
# Running once enables the pre-push PPL regression gate for kernel /
# model / CUDA changes. Re-running is idempotent.

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || {
    echo "Not inside a git repo." >&2
    exit 1
})

cd "$repo_root"

if [ ! -d .githooks ]; then
    echo "No .githooks/ directory at $repo_root." >&2
    exit 1
fi

# Make every hook executable. Windows git-bash ignores the exec bit but
# also runs scripts via their shebang, so this is harmless there.
chmod +x .githooks/* 2>/dev/null || true

git config core.hooksPath .githooks

echo "Hooks installed:"
ls -1 .githooks
echo ""
echo "core.hooksPath = $(git config --get core.hooksPath)"
echo ""
echo "Pre-push PPL gate is active for kernel / model / cuda edits."
echo "Bypass one push:     git push --no-verify"
echo "Disable for session: TQ_SKIP_PPL_GATE=1 git push"
