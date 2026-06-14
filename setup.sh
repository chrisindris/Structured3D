#!/bin/bash

module load apptainer
apptainer build ./apptainer/pymesh.sif docker://pymesh/pymesh:latest
apptainer overlay create --fakeroot --size 4096 ./apptainer/overlay.img
apptainer run --overlay ./apptainer/overlay.img --fakeroot -B /etc/pki/tls/certs/ca-bundle.crt ./apptainer/pymesh.sif bash -c "pip install -r requirements.txt"