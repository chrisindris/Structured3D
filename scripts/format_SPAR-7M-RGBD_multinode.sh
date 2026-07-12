#!/bin/bash

set -e
set -o pipefail

# to unzip, pushd the directory which you want as the folder's parent
# tar -xzvf /scratch/indrisch/huggingface/hub/datasets--jasonzhango--SPAR-7M-RGBD/snapshots/60ef8b2df6430524da86757dec86dcbc55708a41/spar-rgbd-00.tar.gz

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

SCENE_TO_USE=""
PRECOUNT_PROGRESS=true
TOTAL_SLURM_CPUS=$((SLURM_NNODES * SLURM_CPUS_PER_TASK))
WORKERS="${TOTAL_SLURM_CPUS:-16}"
NODE_COUNT="${SLURM_NNODES:-1}"
NODE_INDEX="${SLURM_PROCID:-0}"
TAR_LIST_FILE="/scratch/indrisch/spar-rgbd-full-file-list.txt"
RESUME_TAR_GZ="${RESUME_TAR_GZ:-}"
COMBINED_TAR_GZ="/scratch/indrisch/spar-rgbd-full.tar.gz"
FINAL_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode_v2"
FINAL_DATASET_TAR_GZ="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode_v2.tar.gz"
OVERWRITE_JSONL=false
SKIP_EXISTING_ARTIFACTS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-precount-progress)
            PRECOUNT_PROGRESS=false
            shift
            ;;
        --combined-tar-gz)
            COMBINED_TAR_GZ="$2"
            shift 2
            ;;
        --resume-tar-gz)
            RESUME_TAR_GZ="$2"
            shift 2
            ;;
        --tar-list-file)
            PRECOUNT_PROGRESS=true
            TAR_LIST_FILE="$2"
            shift 2
            ;;
        --final-dataset-dir)
            FINAL_DATASET_DIR="$2"
            shift 2
            ;;
        --final-dataset-tar-gz)
            FINAL_DATASET_TAR_GZ="$2"
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
            echo "Usage: $0 [--no-precount-progress] [--resume-tar-gz PATH] [--combined-tar-gz PATH] [--tar-list-file PATH] [--final-dataset-dir PATH] [--final-dataset-tar-gz PATH] [--overwrite-jsonl] [--skip-existing-artifacts] [SCENE_TO_USE]"
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
echo "TOTAL_SLURM_CPUS: $TOTAL_SLURM_CPUS"
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

if [[ "$CLUSTER" == "TRILLIUM" ]]; then
    echo "On Trillium, it is suggested to create venvs on login node in HOME and then source: https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment"
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

# ---===--- 2. Find and process the dataset files ---===---
if [[ -z "$SLURM_TMPDIR" ]]; then
    export WORKDIR="/scratch/indrisch/Structured3D/spar_data_workdir/node_${NODE_INDEX}/"
else
    export WORKDIR="${SLURM_TMPDIR}/spar_data_workdir/node_${NODE_INDEX}/"
fi
mkdir -p "${WORKDIR}"
echo "WORKDIR: ${WORKDIR}"

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
    export COMBINED_TAR_GZ
    echo "COMBINED_TAR_GZ: ${COMBINED_TAR_GZ}" 
else
    COMBINED_TAR_GZ_LOCAL="${SLURM_TMPDIR}/spar-rgbd-full.tar.gz"
    echo "Copying COMBINED_TAR_GZ from ${COMBINED_TAR_GZ} to ${COMBINED_TAR_GZ_LOCAL}"
    cp "${COMBINED_TAR_GZ}" "${COMBINED_TAR_GZ_LOCAL}"
    export COMBINED_TAR_GZ="${COMBINED_TAR_GZ_LOCAL}"
    echo "COMBINED_TAR_GZ: ${COMBINED_TAR_GZ}" 
fi

if [[ -n "$RESUME_TAR_GZ" ]]; then
    SKIP_EXISTING_ARTIFACTS=true
fi


TAR_LIST_FILE_LOCAL="${WORKDIR}/spar-rgbd-full-file-list.txt"
echo "TAR_LIST_FILE: ${TAR_LIST_FILE}"
echo "TAR_LIST_FILE_LOCAL: ${TAR_LIST_FILE_LOCAL}"

if [[ "$PRECOUNT_PROGRESS" == true ]]; then

    if [[ -f "$TAR_LIST_FILE" ]]; then
        if [[ "$TAR_LIST_FILE" != "$TAR_LIST_FILE_LOCAL" ]]; then
            echo "Copying TAR_LIST_FILE: ${TAR_LIST_FILE} to TAR_LIST_FILE_LOCAL: ${TAR_LIST_FILE_LOCAL}"
            cp "$TAR_LIST_FILE" "$TAR_LIST_FILE_LOCAL"
        else
            echo "Using existing TAR_LIST_FILE_LOCAL: ${TAR_LIST_FILE_LOCAL}"
        fi
    else
        echo "Building TAR_LIST_FILE_LOCAL: ${TAR_LIST_FILE_LOCAL} from COMBINED_TAR_GZ: ${COMBINED_TAR_GZ}"
        if command -v pigz >/dev/null 2>&1; then
            tar --ignore-zeros -I "pigz -dc -p ${WORKERS}" -tf "${COMBINED_TAR_GZ}" > "${TAR_LIST_FILE_LOCAL}"
        else
            tar --ignore-zeros -ztf "${COMBINED_TAR_GZ}" > "${TAR_LIST_FILE_LOCAL}"
        fi
    fi

    REGULAR_FILE_TOTAL=$(grep -vc '/$' "${TAR_LIST_FILE_LOCAL}")
    echo "REGULAR_FILE_TOTAL: ${REGULAR_FILE_TOTAL}"
else
    if [[ -z "$SLURM_TMPDIR" ]]; then
        rm -f "${TAR_LIST_FILE_LOCAL}"
    fi
    echo "Progress pre-scan disabled"
fi


if [[ -z "$SLURM_TMPDIR" ]]; then
    export COMBINED_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode_v2/"
    echo "COMBINED_DATASET_DIR: ${COMBINED_DATASET_DIR}" 
else
    export COMBINED_DATASET_DIR="${SLURM_TMPDIR}/SPAR-7M-RGBD_data_combined_h5_multinode_v2/"
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
)
if [[ "$OVERWRITE_JSONL" == true ]]; then
    PYTHON_ARGS+=(--overwrite-jsonl)
fi
if [[ "$SKIP_EXISTING_ARTIFACTS" == true ]]; then
    PYTHON_ARGS+=(--skip-existing-artifacts)
fi
if [[ "$PRECOUNT_PROGRESS" == true ]]; then
    PYTHON_ARGS+=(--tar-list-file "${TAR_LIST_FILE_LOCAL}")
fi
"${PYTHON_ARGS[@]}"

# put onto permanent storage
if [[ "${SPAR7M_SKIP_FINAL_PACKAGING:-0}" == "1" ]]; then
    echo "Skipping final packaging in worker step."
else
    echo "Node ${NODE_INDEX}: file count in COMBINED_DATASET_DIR ${COMBINED_DATASET_DIR}: $(find "${COMBINED_DATASET_DIR}" -type f | wc -l)"
    echo "Node ${NODE_INDEX}: disk usage of COMBINED_DATASET_DIR ${COMBINED_DATASET_DIR}: $(du -sh "${COMBINED_DATASET_DIR}")"

    echo "Node ${NODE_INDEX}: rsyncing ${COMBINED_DATASET_DIR} to permanent storage ${FINAL_DATASET_DIR}"
    mkdir -p "${FINAL_DATASET_DIR}"
    rsync -auzh --no-p --no-g "${COMBINED_DATASET_DIR%/}/" "${FINAL_DATASET_DIR}/"

    echo "Node ${NODE_INDEX}: rsync complete."
fi