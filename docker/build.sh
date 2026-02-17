#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build \
  -t orchard-ml:latest \
  -f "$REPO_ROOT/docker/Dockerfile" \
  "$REPO_ROOT"
