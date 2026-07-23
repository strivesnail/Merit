#!/usr/bin/env bash
# Moved to experiments/test-scripts/ — thin forwarder.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../test-scripts" && pwd)/run_sift1m_k5_compare.sh" "$@"
