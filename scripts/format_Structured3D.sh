#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-00:10:00
#SBATCH --mem=128GB
#SBATCH --output=out/%N-format_Structured3D-%j.out
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

set -euo pipefail

# Extract each panorama, perspective_empty, or perspective_full archive into WORKDIR, ingest into
# a combined store, and clear WORKDIR between archives to avoid creating
# millions of tiny files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_WORKERS="${SLURM_CPUS_PER_TASK:-32}"
WORKERS="${DEFAULT_WORKERS}"
SCENE_GLOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --scene-glob)
            SCENE_GLOB="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--workers N] [--scene-glob PATTERN]"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

DATA_ROOT="/scratch/indrisch/Structured3D/data"
FINAL_COMBINED_DATASET="/scratch/indrisch/Structured3D_data_combined"

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    USE_FAST_TMP=false
    export WORKDIR="/scratch/indrisch/Structured3D/workdir_structured3d"
    export COMBINED_DATASET="${FINAL_COMBINED_DATASET}"
    INPUT_STAGE_DIR=""
else
    USE_FAST_TMP=true
    export WORKDIR="${SLURM_TMPDIR}/workdir_structured3d"
    export COMBINED_DATASET="${SLURM_TMPDIR}/Structured3D_data_combined"
    INPUT_STAGE_DIR="${SLURM_TMPDIR}/structured3d_input"
    mkdir -p "${INPUT_STAGE_DIR}"
fi

mkdir -p "${WORKDIR}" "${COMBINED_DATASET}" "${SCRIPT_DIR}/out"

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    export VENV_SPAR7M="/scratch/indrisch/venv_spar7m"
    source "${VENV_SPAR7M}/bin/activate"
else
    BASE_VENV="/scratch/indrisch/venv_spar7m"
    export VENV_SPAR7M="${SLURM_TMPDIR}/venv_spar7m"
    rm -rf "${VENV_SPAR7M}"
    cp -a "${BASE_VENV}" "${VENV_SPAR7M}"
    source "${VENV_SPAR7M}/bin/activate"
fi

if [[ "${USE_FAST_TMP}" == true && -d "${FINAL_COMBINED_DATASET}" ]]; then
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${FINAL_COMBINED_DATASET}/" "${COMBINED_DATASET}/"
    else
        cp -a "${FINAL_COMBINED_DATASET}/." "${COMBINED_DATASET}/"
    fi
fi

clear_workdir() {
    find "${WORKDIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

clear_workdir

if [[ -n "${SCENE_GLOB}" ]]; then
    FIND_ARGS=(-type f -name "${SCENE_GLOB}")
else
    FIND_ARGS=(-type f \( -name "*panorama*.zip" -o -name "*perspective_empty*.zip" -o -name "*perspective_full*.zip" \))
fi

while IFS= read -r -d '' CURRENT_ZIP; do
    echo "${CURRENT_ZIP}"
    clear_workdir
    ZIP_TO_EXTRACT="${CURRENT_ZIP}"
    if [[ "${USE_FAST_TMP}" == true ]]; then
        ZIP_TO_EXTRACT="${INPUT_STAGE_DIR}/$(basename "${CURRENT_ZIP}")"
        cp "${CURRENT_ZIP}" "${ZIP_TO_EXTRACT}"
    fi

    ~/7zz x "${ZIP_TO_EXTRACT}" -o"${WORKDIR}"
    python "${SCRIPT_DIR}/format_Structured3D.py" \
        --combined-dataset "${COMBINED_DATASET}" \
        --curr-dataset "${WORKDIR}" \
        --input-tar-gz "${CURRENT_ZIP}" \
        --workers "${WORKERS}"

    if [[ "${USE_FAST_TMP}" == true ]]; then
        rm -f "${ZIP_TO_EXTRACT}"
    fi

    clear_workdir
done < <(find "${DATA_ROOT}" "${FIND_ARGS[@]}" -print0 | sort -z)

if [[ "${USE_FAST_TMP}" == true ]]; then
    mkdir -p "${FINAL_COMBINED_DATASET}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${COMBINED_DATASET}/" "${FINAL_COMBINED_DATASET}/"
    else
        cp -a "${COMBINED_DATASET}/." "${FINAL_COMBINED_DATASET}/"
    fi
fi