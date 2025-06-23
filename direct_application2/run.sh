#!/bin/bash

mkdir -p ./fwd_res
mkdir -p ./outputs
mkdir -p ./pred_subfiles
mkdir -p ./catalogs
mkdir -p /scratch/aanthore/LADUMA_data/WORK2

if [ $# -lt 1 ]; then
    echo -e "ERROR: Requires network for inference."
    echo -e "Expected: $0 network.dat list_data.txt"
    exit 1
fi

python3 -W "ignore" main.py $1 $2 1> ./outputs/out1 2> ./outputs/out2 &

exit 0
