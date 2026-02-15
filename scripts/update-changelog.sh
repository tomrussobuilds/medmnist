#!/usr/bin/env bash
# Regenerates the [Unreleased] section of CHANGELOG.md using git-cliff.
# Called automatically by the pre-commit hook.
set -euo pipefail

CHANGELOG="CHANGELOG.md"

# Generate only the unreleased section (no header, no past releases)
unreleased=$(git cliff --unreleased --strip header 2>/dev/null || true)

# If there's nothing unreleased, keep just the empty header
if [ -z "$unreleased" ] || [ "$unreleased" = "## [Unreleased]" ]; then
    unreleased="## [Unreleased]"
fi

# Extract the static part: everything from the first versioned release onward
static=$(sed -n '/^## \[[0-9]/,$p' "$CHANGELOG")

# Rebuild the changelog
cat > "$CHANGELOG" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

$unreleased

$static
EOF

git add "$CHANGELOG"
