#!/usr/bin/env bash
# Accuracy CSV for one xpmath box backend, split the way plot_quad_accuracy.py
# expects: result_xpmath/<backend>/<INTEGRAL>_val.txt, no header.
#
# Usage: scripts/run_xpmath_box_accuracy.sh dd [batch] [variant]
# batch defaults to 100000, matching Box_true.txt.
# variant, when set, writes result_xpmath/<variant>/<backend>/ instead of
# result_xpmath/<backend>/. BUILD_DIR selects the binaries.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="${1:?backend: dd, ff, qf, or tf}"
BATCH="${2:-100000}"
VARIANT="${3:-}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/serial-5.1.0}"
EXE="${BUILD_DIR}/boxGPU_test_${BACKEND}"
if [[ -n "${VARIANT}" ]]; then
    OUT_DIR="${REPO_ROOT}/result_xpmath/${VARIANT}/${BACKEND}"
else
    OUT_DIR="${REPO_ROOT}/result_xpmath/${BACKEND}"
fi
RAW="${OUT_DIR}/all.csv"

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: ${EXE} not found" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
echo "Running ${EXE} 1 ${BATCH} -> ${OUT_DIR}/"
"${EXE}" 1 "${BATCH}" > "${RAW}"

# Drop the three preamble lines (mode, batch, column header).
tail -n +4 "${RAW}" | awk -F, -v dir="${OUT_DIR}" '
    $1 != "" { print > (dir "/" $1 "_val.txt") }
'
rm -f "${RAW}"

echo "Done. Files in ${OUT_DIR}/:"
wc -l "${OUT_DIR}"/*_val.txt
