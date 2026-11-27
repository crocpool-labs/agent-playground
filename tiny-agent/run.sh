#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ "${1:-}" = "setup" ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install --no-cache-dir -r requirements.txt -c requirements.lock.txt
    exit 0
fi
if [ ! -x .venv/bin/python ]; then
    echo '请先运行：sh run.sh setup' >&2
    exit 1
fi
exec .venv/bin/python -m tiny_agent "$@"
