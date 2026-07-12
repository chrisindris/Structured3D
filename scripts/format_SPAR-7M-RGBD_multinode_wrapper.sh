#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-03:00:00
#SBATCH --mem=0
#SBATCH --output=out/%N-format_SPAR-7M-RGBD_multinode-%j.out
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

set -e
set -o pipefail

TAR_LIST_FILE="/scratch/indrisch/spar-rgbd-full-file-list.txt"
RESUME_TAR_GZ="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_partial.tar.gz"

srun \
	--ntasks="${SLURM_NNODES:-1}" \
	--ntasks-per-node=1 \
	--cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
	env SPAR7M_SKIP_FINAL_PACKAGING=0 \
	/scratch/indrisch/Structured3D/scripts/format_SPAR-7M-RGBD_multinode.sh \
	--overwrite-jsonl \
	--skip-existing-artifacts \
	--tar-list-file "${TAR_LIST_FILE}" \
	--resume-tar-gz "${RESUME_TAR_GZ}"

FINAL_DATASET_DIR="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode_v2"
FINAL_DATASET_TAR_GZ="/scratch/indrisch/SPAR-7M-RGBD_data_combined_h5_multinode_v2.tar.gz"
if [[ -e "${FINAL_DATASET_TAR_GZ}" ]]; then
	echo "Error: Final tar archive already exists: ${FINAL_DATASET_TAR_GZ}" >&2
	echo "Remove or rename it before rerunning to avoid overwriting a completed run." >&2
fi

echo "file count in FINAL_DATASET_DIR: $(find "${FINAL_DATASET_DIR}" -type f | wc -l)"
echo "disk usage of FINAL_DATASET_DIR: $(du -sh "${FINAL_DATASET_DIR}")"

# echo "Store a tar.gz on scratch at ${FINAL_DATASET_TAR_GZ} (TODO: transfer to nearline; it seems as though this cannot be done through compute nodes)"
# tar -czf "${FINAL_DATASET_TAR_GZ}" "${FINAL_DATASET_DIR}"