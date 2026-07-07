#!/bin/bash

set -e
set -o pipefail

# to unzip, pushd the directory which you want as the folder's parent
# tar -xzvf /scratch/indrisch/huggingface/hub/datasets--jasonzhango--SPAR-7M-RGBD/snapshots/60ef8b2df6430524da86757dec86dcbc55708a41/spar-rgbd-00.tar.gz

# ---===--- 1. Set up the Python environment ---===---

SCENE_TO_USE=""
PRECOUNT_PROGRESS=true
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
NODE_COUNT="${SLURM_NNODES:-1}"
NODE_INDEX="${SLURM_PROCID:-0}"
RESUME_TAR_GZ="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_partial.tar.gz"
OVERWRITE_JSONL=false
SKIP_EXISTING_ARTIFACTS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-precount-progress)
            PRECOUNT_PROGRESS=false
            shift
            ;;
        --resume-tar-gz)
            RESUME_TAR_GZ="$2"
            shift 2
            ;;
        --overwrite-jsonl)
            OVERWRITE_JSONL=true
            shift
            ;;
        --skip-existing-artifacts)
            SKIP_EXISTING_ARTIFACTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-precount-progress] [--resume-tar-gz PATH] [--overwrite-jsonl] [--skip-existing-artifacts] [SCENE_TO_USE]"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: Unknown flag '$1'"
            exit 1
            ;;
        *)
            if [[ -z "$SCENE_TO_USE" ]]; then
                SCENE_TO_USE="$1"
                shift
            else
                echo "Error: Unexpected extra positional argument '$1'"
                exit 1
            fi
            ;;
    esac
done
echo "SCENE_TO_USE: $SCENE_TO_USE"
echo "PRECOUNT_PROGRESS: $PRECOUNT_PROGRESS"
echo "WORKERS: $WORKERS"
echo "NODE_COUNT: $NODE_COUNT"
echo "NODE_INDEX: $NODE_INDEX"
echo "RESUME_TAR_GZ: $RESUME_TAR_GZ"
echo "OVERWRITE_JSONL: $OVERWRITE_JSONL"
echo "SKIP_EXISTING_ARTIFACTS: $SKIP_EXISTING_ARTIFACTS"

if [[ "$PWD" == *Structured3D* ]]; then
    PROJECT_DIR="${PWD%%Structured3D*}/Structured3D"
else
    echo "Error: Could not find 'Structured3D' in the current path."
    exit 1
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="$PYTHONPATH:$SYSCONFIG_DIR_PATH"

echo "PROJECT_DIR: $PROJECT_DIR"
echo "SYSCONFIG_DIR_PATH: $SYSCONFIG_DIR_PATH"
echo "PWD: $PWD"
echo "PYTHONPATH: $PYTHONPATH"

export HF_DEBUG=1
HF_TOKEN=$(cat /home/indrisch/TOKENS/cvis-tmu-organization-token.txt)
export HF_TOKEN

# vLLM models typically come from the huggingface hub. 
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

if [[ -z "$SLURM_TMPDIR" ]]; then
    export VENV_SPAR7M="/scratch/indrisch/venv_spar7m/" 
    source ${VENV_SPAR7M}/bin/activate
else
    export VENV_SPAR7M="${SLURM_TMPDIR}/venv_spar7m/" 
    virtualenv --no-download ${VENV_SPAR7M}
    source ${VENV_SPAR7M}/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install numpy torch pyarrow h5py opencv-python huggingface_hub tqdm
fi
echo "Venv path: ${VENV_SPAR7M}"

# ---===--- 2. Find and process the dataset files ---===---
if [[ -z "$SLURM_TMPDIR" ]]; then
    export WORKDIR="/scratch/indrisch/Structured3D/spar_data_workdir/node_${NODE_INDEX}/"
else
    export WORKDIR="${SLURM_TMPDIR}/spar_data_workdir/node_${NODE_INDEX}/"
fi
mkdir -p "${WORKDIR}"
echo "WORKDIR: ${WORKDIR}"

TAR_LIST_FILE="${WORKDIR}/spar-rgbd-full-file-list.txt"
echo "TAR_LIST_FILE: ${TAR_LIST_FILE}"

# if SCENE_TO_USE is unset, set CURR_SCENE="spar-rgbd-*.tar.gz"; otherwise, set CURR_SCENE="spar-rgbd-${SCENE_TO_USE}.tar.gz"
if [[ -z "$SCENE_TO_USE" ]]; then
    CURR_SCENE="spar-rgbd-*.tar.gz"
else
    # make SCENE_TO_USE two digits with leading zeros if necessary
    SCENE_TO_USE=$(printf "%02d" "$SCENE_TO_USE")
    CURR_SCENE="spar-rgbd-${SCENE_TO_USE}.tar.gz"
fi
echo "CURR_SCENE: $CURR_SCENE"

if [[ -n "$SCENE_TO_USE" ]]; then
    echo "Note: SCENE_TO_USE is currently ignored in streaming COMBINED_TAR_GZ mode."
fi


if [[ -z "$SLURM_TMPDIR" ]]; then
    export COMBINED_TAR_GZ="/scratch/indrisch/spar-rgbd-full.tar.gz"
    echo "COMBINED_TAR_GZ: ${COMBINED_TAR_GZ}" 
else
    COMBINED_TAR_GZ="${SLURM_TMPDIR}/spar-rgbd-full.tar.gz"
    cp "/scratch/indrisch/spar-rgbd-full.tar.gz" "${COMBINED_TAR_GZ}"
    echo "COMBINED_TAR_GZ: ${COMBINED_TAR_GZ}" 
fi

if [[ "$PRECOUNT_PROGRESS" == true ]]; then
    if command -v pigz >/dev/null 2>&1; then
        tar --ignore-zeros -I "pigz -dc -p ${WORKERS}" -tf "${COMBINED_TAR_GZ}" > "${TAR_LIST_FILE}"
    else
        tar --ignore-zeros -ztf "${COMBINED_TAR_GZ}" > "${TAR_LIST_FILE}"
    fi
    REGULAR_FILE_TOTAL=$(grep -vc '/$' "${TAR_LIST_FILE}")
    echo "REGULAR_FILE_TOTAL: ${REGULAR_FILE_TOTAL}"
else
    rm -f "${TAR_LIST_FILE}"
    echo "Progress pre-scan disabled"
fi


if [[ -z "$SLURM_TMPDIR" ]]; then
    export COMBINED_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode/"
    echo "COMBINED_DATASET_DIR: ${COMBINED_DATASET_DIR}" 
else
    export COMBINED_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode/"
    echo "COMBINED_DATASET_DIR: ${COMBINED_DATASET_DIR}" 
fi

if [[ -n "$RESUME_TAR_GZ" ]]; then
    if [[ -z "$SLURM_TMPDIR" ]]; then
        echo "Error: --resume-tar-gz requires SLURM_TMPDIR to be set." >&2
        exit 1
    fi
    if [[ ! -f "$RESUME_TAR_GZ" ]]; then
        echo "Error: Resume archive not found: $RESUME_TAR_GZ" >&2
        exit 1
    fi

    RESUME_TAR_COPY="${SLURM_TMPDIR}/$(basename "$RESUME_TAR_GZ")"
    cp "$RESUME_TAR_GZ" "$RESUME_TAR_COPY"
    echo "RESUME_TAR_COPY: ${RESUME_TAR_COPY}"

    if command -v pigz >/dev/null 2>&1; then
        tar -I "pigz -dc -p ${WORKERS}" -xf "$RESUME_TAR_COPY" -C "$SLURM_TMPDIR" --strip-components=2
    else
        tar -xzf "$RESUME_TAR_COPY" -C "$SLURM_TMPDIR" --strip-components=2
    fi
fi

# find /scratch/indrisch/huggingface/hub/datasets--jasonzhango--SPAR-7M-RGBD/snapshots/60ef8b2df6430524da86757dec86dcbc55708a41/ \
#   -name "$CURR_SCENE" \
#   -exec sh -c '
#     pushd "${WORKDIR}";
#     tar -xzvf "$1";
#     popd;
#     python format_SPAR-7M-RGBD_multinode.py \
#         --combined-dataset "${COMBINED_DATASET_DIR}" \
#         --curr-dataset "${WORKDIR}/spar/";
#     rm -rf "${WORKDIR}/spar/"
#   ' _ {} \;

PYTHON_ARGS=(
    python format_SPAR-7M-RGBD_multinode.py
    --combined-dataset "${COMBINED_DATASET_DIR}"
    --input-tar-gz "${COMBINED_TAR_GZ}"
    --extract-root "${WORKDIR}"
    --workers "${WORKERS}"
    --node-count "${NODE_COUNT}"
    --node-index "${NODE_INDEX}"
    --overwrite-jsonl
    --skip-existing-artifacts
    --tar-list-file "/scratch/indrisch/spar-rgbd-full-file-list.txt"
)
if [[ "$OVERWRITE_JSONL" == true ]]; then
    PYTHON_ARGS+=(--overwrite-jsonl)
fi
if [[ "$SKIP_EXISTING_ARTIFACTS" == true ]]; then
    PYTHON_ARGS+=(--skip-existing-artifacts)
fi
if [[ "$PRECOUNT_PROGRESS" == true ]]; then
    PYTHON_ARGS+=(--tar-list-file "${TAR_LIST_FILE}")
fi
"${PYTHON_ARGS[@]}"

# put onto permanent storage
FINAL_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode"
FINAL_DATASET_TAR_GZ="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode.tar.gz"
if [[ "${SPAR7M_SKIP_FINAL_PACKAGING:-0}" == "1" ]]; then
    echo "Skipping final packaging in worker step."
elif [[ "${NODE_INDEX}" == "0" ]]; then
    if [[ -e "${FINAL_DATASET_TAR_GZ}" ]]; then
        echo "Error: Final tar archive already exists: ${FINAL_DATASET_TAR_GZ}" >&2
        echo "Remove or rename it before rerunning to avoid overwriting a completed run." >&2
        exit 1
    fi

    echo "file count in COMBINED_DATASET_DIR: $(find "${COMBINED_DATASET_DIR}" -type f | wc -l)"
    echo "disk usage of COMBINED_DATASET_DIR: $(du -sh "${COMBINED_DATASET_DIR}")"

    echo "Store a tar.gz on scratch at ${FINAL_DATASET_TAR_GZ} (TODO: transfer to nearline; it seems as though this cannot be done through compute nodes)"
    tar -czf "${FINAL_DATASET_TAR_GZ}" "${COMBINED_DATASET_DIR}"
else
    echo "Skipping final packaging on node ${NODE_INDEX}; node 0 will handle it."
fi