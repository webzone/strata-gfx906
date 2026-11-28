#!/usr/bin/env bash
# Run the fleet check interpreter with an isolated HOME/XDG/TMPDIR, so a test never touches the host's own
# Strata install or data. The venv is ~/fleet-envs/strata-gfx906 (tools/fleet/prepare.sh).
set -euo pipefail
py="$HOME/fleet-envs/strata-gfx906/bin/python"
[ -x "$py" ] || { echo "py.sh: no toolchain at $py (run fleet-check prepare)" >&2; exit 2; }
iso=$(mktemp -d "${TMPDIR:-/tmp}/strata-check.XXXXXX")
trap 'rm -rf "$iso"' EXIT
mkdir -p "$iso/home" "$iso/cfg" "$iso/data" "$iso/state" "$iso/tmp"
export STRATA_CHECK_REAL_HOME="$HOME"
HOME="$iso/home" XDG_CONFIG_HOME="$iso/cfg" XDG_DATA_HOME="$iso/data" XDG_STATE_HOME="$iso/state" TMPDIR="$iso/tmp" \
  "$py" "$@"
