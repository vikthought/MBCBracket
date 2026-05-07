#!/usr/bin/env bash
# check_embedded_run.sh — is run_real_embedded.py still going?
#
# Usage:
#   ./check_embedded_run.sh            # one-shot status
#   ./check_embedded_run.sh wait       # block until it finishes, then exit 0
#
# When done, run:  .venv/bin/python analyze_real_embedded.py

set -uo pipefail

is_running() {
    pgrep -f "run_real_embedded.py" >/dev/null 2>&1
}

count_rows() {
    local f="results/real_embedded/real_embedded_raw.csv"
    if [ -f "$f" ]; then
        # subtract 1 for the header
        echo $(($(wc -l < "$f") - 1))
    else
        echo 0
    fi
}

print_status() {
    if is_running; then
        local pid
        pid=$(pgrep -f "run_real_embedded.py" | head -n1)
        echo "RUNNING  pid=$pid  rows so far: $(count_rows)"
    else
        echo "DONE     rows: $(count_rows)"
        echo "Run:  .venv/bin/python analyze_real_embedded.py"
    fi
}

case "${1:-status}" in
    status)
        print_status
        ;;
    wait)
        echo "Polling every 30s. Ctrl-C to stop watching (the run continues regardless)."
        while is_running; do
            echo "  $(date '+%H:%M:%S')  rows so far: $(count_rows)"
            sleep 30
        done
        echo "DONE  rows: $(count_rows)"
        echo "Run:  .venv/bin/python analyze_real_embedded.py"
        ;;
    *)
        echo "Usage: $0 [status|wait]" >&2
        exit 1
        ;;
esac
