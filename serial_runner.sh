#!/usr/bin/bash

# run this script in a screen session.
# this script will run sbatch in parallel; it will request a job, and then check every 10 minutes to see if the job has completed (i.e. sq | wc -l returns 1); once the job has completed, it will run sbatch again, until all jobs have been submitted.

for i in {1..17}; do
    echo "Submitting job $i"
    pushd "/scratch/indrisch/Structured3D/data_sandbox/" || exit
    sbatch format_SPAR-7M-RGBD.sh "$i"
    popd || exit
    echo "Job $i submitted. Waiting for completion..."
    sleep 600  # wait for 10 minutes before checking the job status
    while [ "$(sq | wc -l)" -gt 1 ]; do
        echo "Job $i hasn't finished. Checking again in 10 minutes..."
        sleep 600
    done
    echo "Job $i has completed."
done