#!/bin/bash

set -e
set -o pipefail

# ---===--- 1. Set up the Python environment ---===---

# Detect cluster based on terminal prompt or hostname
if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$PS1" == *"tri"* ]] || [[ "$HOSTNAME" == *"tri"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi
echo "Detected cluster: $CLUSTER"

NODE_COUNT="${SLURM_NNODES:-1}"
NODE_INDEX="${SLURM_PROCID:-0}"
INPUT_TAR_GZ=""
FINAL_DATASET_DIR="/scratch/indrisch/Structured3D_data_combined_h5_multinode"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-tar-gz)
            INPUT_TAR_GZ="$2"
            shift 2
            ;;
        --final-dataset-dir)
            FINAL_DATASET_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_TAR_GZ" ]]; then
    echo "Error: --input-tar-gz is required."
    exit 1
fi

if [[ "$PWD" == *Structured3D* ]]; then
    PROJECT_DIR="${PWD%%Structured3D*}/Structured3D"
else
    echo "Error: Could not find 'Structured3D' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

# Load modules
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

if [[ "$CLUSTER" == "TRILLIUM" ]]; then
    export VENV_SPAR7M="/home/indrisch/venv_spar7m/"
    source ${VENV_SPAR7M}/bin/activate
else
    if [[ -z "$SLURM_TMPDIR" ]]; then
        export VENV_SPAR7M="/scratch/indrisch/venv_spar7m/" 
        source ${VENV_SPAR7M}/bin/activate
    else
        export VENV_SPAR7M="${SLURM_TMPDIR}/venv_spar7m/" 
        virtualenv --no-download ${VENV_SPAR7M}
        source ${VENV_SPAR7M}/bin/activate
        pip install --no-index --upgrade pip setuptools wheel
        pip install --no-index numpy torch pyarrow h5py opencv-python huggingface_hub tqdm
    fi
fi
echo "Venv path: ${VENV_SPAR7M}"

# ---===--- 2. Extract and assign ZIP files ---===---

if [[ -z "$SLURM_TMPDIR" ]]; then
    export WORKDIR="/scratch/indrisch/Structured3D_workdir/node_${NODE_INDEX}/"
    export COMBINED_DATASET_DIR="/scratch/indrisch/Structured3D_data_combined_h5_multinode_temp/"
else
    export WORKDIR="${SLURM_TMPDIR}/Structured3D_workdir/node_${NODE_INDEX}/"
    export COMBINED_DATASET_DIR="${SLURM_TMPDIR}/Structured3D_data_combined_h5_multinode_temp/"
fi
mkdir -p "${WORKDIR}"
mkdir -p "${COMBINED_DATASET_DIR}"
echo "WORKDIR: ${WORKDIR}"

if [[ -z "$SLURM_TMPDIR" ]]; then
    TAR_GZ_LOCAL="${INPUT_TAR_GZ}"
else
    TAR_GZ_LOCAL="${SLURM_TMPDIR}/$(basename "${INPUT_TAR_GZ}")"
    echo "Copying ${INPUT_TAR_GZ} to ${TAR_GZ_LOCAL}"
    cp "${INPUT_TAR_GZ}" "${TAR_GZ_LOCAL}"
fi

# Extract the .tar.gz to get the .zip/.7z files
echo "Extracting ${TAR_GZ_LOCAL} to ${WORKDIR}/zips"
mkdir -p "${WORKDIR}/zips"
tar -xf "${TAR_GZ_LOCAL}" -C "${WORKDIR}/zips"

# Find all zip/7z files
shopt -s nullglob
ZIP_FILES=($(find "${WORKDIR}/zips" -type f \( -name "*.zip" -o -name "*.7z" \) | sort))

# Assign to current node
ASSIGNED_ZIPS=()
for i in "${!ZIP_FILES[@]}"; do
    if (( i % NODE_COUNT == NODE_INDEX )); then
        ASSIGNED_ZIPS+=("${ZIP_FILES[$i]}")
    fi
done

echo "Node $NODE_INDEX assigned ${#ASSIGNED_ZIPS[@]} zip files out of ${#ZIP_FILES[@]} total."

EXTRACT_ROOT="${WORKDIR}/extracted"
mkdir -p "${EXTRACT_ROOT}"

# Extract assigned zip/7z files
for zf in "${ASSIGNED_ZIPS[@]}"; do
    echo "Extracting $zf..."
    if [[ -f ~/7zz ]]; then
        ~/7zz x "$zf" -o"${EXTRACT_ROOT}" -y >/dev/null
    else
        echo "Error: ~/7zz not found. Please install 7-zip."
        exit 1
    fi
done

echo "Extraction complete."

# ---===--- 3. Run Python formatter ---===---

python format_Structured3D_multinode.py \
    --combined-dataset "${COMBINED_DATASET_DIR}" \
    --extract-root "${EXTRACT_ROOT}"

# ---===--- 4. Rsync to final destination ---===---

if [[ "${SPAR7M_SKIP_FINAL_PACKAGING:-0}" == "1" ]]; then
    echo "Skipping final packaging in worker step."
else
    echo "Node ${NODE_INDEX}: rsyncing ${COMBINED_DATASET_DIR} to permanent storage ${FINAL_DATASET_DIR}"
    mkdir -p "${FINAL_DATASET_DIR}"
    rsync -auzh --no-p --no-g "${COMBINED_DATASET_DIR%/}/" "${FINAL_DATASET_DIR}/"
    echo "Node ${NODE_INDEX}: rsync complete."
fi
