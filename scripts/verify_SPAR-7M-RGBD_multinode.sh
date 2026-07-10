#!/bin/bash
# Wrapper script to verify SPAR-7M-RGBD multi-node HDF5/JSON conversion.
# Complies with AllianceCan / Compute Canada environment & wheelhouse rules.

set -e
set -o pipefail

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

# Resolve project root
if [[ "$PWD" == *Structured3D* ]]; then
    PROJECT_DIR="${PWD%%Structured3D*}/Structured3D"
else
    PROJECT_DIR="$PWD"
fi
SYSCONFIG_DIR_PATH="$PROJECT_DIR/scripts"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SYSCONFIG_DIR_PATH"

# Load AllianceCan modules BEFORE activating virtualenv
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow scipy-stack

if [[ "$CLUSTER" == "TRILLIUM" ]]; then
    export VENV_SPAR7M="/home/indrisch/venv_spar7m/"
    source "${VENV_SPAR7M}/bin/activate"
else
    if [[ -z "$SLURM_TMPDIR" ]]; then
        export VENV_SPAR7M="/scratch/indrisch/venv_spar7m/"
        source "${VENV_SPAR7M}/bin/activate"
    else
        export VENV_SPAR7M="${SLURM_TMPDIR}/venv_spar7m/"
        virtualenv --no-download "${VENV_SPAR7M}"
        source "${VENV_SPAR7M}/bin/activate"
        pip install --no-index --upgrade pip setuptools wheel
        pip install --no-index numpy h5py pillow pytest tqdm
    fi
fi
echo "Activated venv: ${VENV_SPAR7M}"

# Check mode
RUN_PYTEST=false
PYTEST_ARGS=()
CLI_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pytest)
            RUN_PYTEST=true
            shift
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p out

if [[ "$RUN_PYTEST" == "true" ]]; then
    echo "Running pytest verification suite..."
    python3 -m pytest scripts/test_verify_SPAR_7M_RGBD.py -v "${CLI_ARGS[@]}" 2>&1 | tee /scratch/indrisch/Structured3D/out/pytest_output.log
else
    echo "Running SPAR-7M-RGBD Multi-Node verification report card engine..."
    python3 scripts/verify_SPAR-7M-RGBD_multinode.py "${CLI_ARGS[@]}"
fi
