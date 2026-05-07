#!/usr/bin/env bash
# run_all.sh — run all three MBC benchmark suites.
#
# Usage:
#   ./run_all.sh                  # synth + neuro + real, defaults
#   ./run_all.sh synth            # just synth
#   ./run_all.sh neuro real       # neuro then real
#   ./run_all.sh --dtm off all    # forward extra args to each runner
#
# Args before the first suite name are forwarded to every runner. Suite names
# come last. Defaults to "all" (= synth neuro real) if no suite is named.
#
# Override Python with PY env var:
#   PY=/opt/homebrew/bin/python3.10 ./run_all.sh
#
set -uo pipefail

cd "$(dirname "$0")"

# Default to the project venv if it exists; the user can override with
# PY=/path/to/python ./run_all.sh
if [ -z "${PY:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    else
        PY="python3"
    fi
fi

# Tee all output to run_all.log in the project root so the user can `tail -f`
# to watch progress without keeping the foreground attached.
LOG="run_all.log"
exec > >(tee "$LOG") 2>&1
echo "=== run_all.sh started: $(date -Iseconds)  PY=$PY ==="
START_TS=$(date +%s)

# Pick out the suite names from the args; everything else passes through.
EXTRA=()
SUITES=()
for arg in "$@"; do
    case "$arg" in
        synth|real|neuro|all) SUITES+=("$arg") ;;
        *) EXTRA+=("$arg") ;;
    esac
done

if [ ${#SUITES[@]} -eq 0 ]; then
    SUITES=("all")
fi

# Expand "all" -> synth neuro real (neuro before real because it's faster).
EXPANDED=()
for s in "${SUITES[@]}"; do
    case "$s" in
        all) EXPANDED+=("synth" "neuro" "real") ;;
        *)   EXPANDED+=("$s") ;;
    esac
done

# Smoke test first; bail if MBC.py is broken before burning compute on a
# multi-suite run.
echo "============================================================"
echo "  Smoke test"
echo "============================================================"
if ! "$PY" test_mbc_smoke.py; then
    echo "smoke test failed — aborting" >&2
    exit 1
fi
echo

declare -i FAILED=0
for s in "${EXPANDED[@]}"; do
    SCRIPT="run_${s}.py"
    if [ ! -f "$SCRIPT" ]; then
        echo "$SCRIPT not found, skipping" >&2
        continue
    fi
    echo "============================================================"
    echo "  $s suite ($(date +%H:%M:%S))"
    echo "  $PY $SCRIPT ${EXTRA[*]:-}"
    echo "============================================================"
    if ! "$PY" "$SCRIPT" ${EXTRA[@]+"${EXTRA[@]}"}; then
        echo "[$s] suite failed" >&2
        FAILED+=1
    fi
    echo
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
HH=$((ELAPSED / 3600)); MM=$(((ELAPSED % 3600) / 60)); SS=$((ELAPSED % 60))

if [ "$FAILED" -gt 0 ]; then
    echo "$FAILED suite(s) failed.  total elapsed: ${HH}h${MM}m${SS}s  log: $LOG" >&2
    exit 1
fi

echo "=== run_all.sh done: $(date -Iseconds)  elapsed: ${HH}h${MM}m${SS}s ==="
echo "Outputs under results/{synth,neuro,real}/.  Full log: $LOG"
