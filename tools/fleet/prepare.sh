#!/usr/bin/env bash
# Fleet toolchain for strata-gfx906 checks: a venv with the two packages the Python tests import.
# Usage: prepare.sh <repo> <env_dir>. Installs nothing outside <env_dir>.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: prepare.sh <repo> <env_dir>" >&2; exit 2; }
env_dir=$2
[ -x "$env_dir/bin/python" ] || python3 -m venv "$env_dir"
"$env_dir/bin/python" -m pip install -q --disable-pip-version-check jinja2 regex
"$env_dir/bin/python" -c "import jinja2, regex; print('strata-gfx906 toolchain ok')"
