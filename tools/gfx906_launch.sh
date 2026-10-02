#!/bin/sh
# T5810: one saved model at a time; never stop an existing service to switch.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CONFIG=${1:?usage: gfx906_launch.sh CONFIG [server options]}
shift
case "$CONFIG" in
    /*) ;;
    *) CONFIG="$ROOT/$CONFIG" ;;
esac
cd "$ROOT"
if [ ! -f "$CONFIG" ]; then
    printf 'Model is not prepared: %s\n' "$CONFIG" >&2
    exit 1
fi
command -v flock >/dev/null 2>&1 || { echo 'flock is required; no model started.' >&2; exit 1; }
command -v ss >/dev/null 2>&1 || { echo 'ss is required; no model started.' >&2; exit 1; }
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/model-launch.lock"
if ! flock -n 9; then
    echo 'Another saved model is running. Stop it manually before switching; nothing was stopped or started.' >&2
    exit 1
fi
LISTENERS=$(ss -H -lnt 'sport = :8082')
if [ -n "$LISTENERS" ]; then
    echo 'Port 8082 is occupied. Stop the current service manually before switching; no model started.' >&2
    exit 1
fi
# The private-LAN deployment intentionally has no API key. Do not export an empty key.
unset HSA_OVERRIDE_GFX_VERSION STRATA_API_KEY
# Descriptor 9 keeps the model lock held for the lifetime of this foreground server.
exec "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata \
    --config "$CONFIG" --host 0.0.0.0 --port 8082 "$@"
