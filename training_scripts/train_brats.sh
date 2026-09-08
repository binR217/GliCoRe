#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
: "${GLICORE_BRATS_ROOT:?Set GLICORE_BRATS_ROOT to the preprocessed BraTS task root.}"
: "${RESULTS_FOLDER:?Set RESULTS_FOLDER to the output root.}"

export PYTHONPATH="${PYTHONPATH:-.}"
export glicore_preprocessed="${GLICORE_BRATS_ROOT}"
export glicore_raw_data_base="${GLICORE_BRATS_RAW_ROOT:-${GLICORE_BRATS_ROOT}}"
export PACER_ENABLE=1
export PACER_MODE="${MODE}"
export PACER_CALIBRATION_BATCHES=16

if [[ "${MODE}" == "prepare" ]]; then
  python -m glicore.run.prepare_pacer_context_cache \
    3d_fullres GliCoReTrainerBraTS 3 0
else
  python -m glicore.run.run_training \
    3d_fullres GliCoReTrainerBraTS 3 0
fi

