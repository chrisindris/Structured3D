#!/usr/bin/bash
# Combine all the SPAR-7M-RGBD tar.gz files into a single tar.gz file for easier handling, according to https://github.com/LogosRoboticsGroup/SPAR/issues/3

set -euo pipefail

COMBINED_TAR_GZ="/scratch/indrisch/spar-rgbd-full.tar.gz"

# find /scratch/indrisch/huggingface/hub/datasets--jasonzhango--SPAR-7M-RGBD/snapshots/60ef8b2df6430524da86757dec86dcbc55708a41/ \
#   -name "spar-rgbd-*.tar.gz" \
#   -exec sh -c '
#     echo {} && cat {} > "$COMBINED_TAR_GZ"
#   ' _ {} \;

pushd "/scratch/indrisch/huggingface/hub/datasets--jasonzhango--SPAR-7M-RGBD/snapshots/60ef8b2df6430524da86757dec86dcbc55708a41/" || exit
cat spar-rgbd-*.tar.gz > "${COMBINED_TAR_GZ}"
popd || exit

echo "Combined archive created at: ${COMBINED_TAR_GZ}"
echo "Use format_SPAR-7M-RGBD.sh to stream-ingest this archive into H5/JSON outputs without bulk extraction."