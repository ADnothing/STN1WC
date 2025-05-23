#!/bin/bash

mkdir -p ./fwd_res
mkdir -p ./outputs
mkdir -p ./pred_subfiles
mkdir -p ./catalogs
mkdir -p /scratch/aanthore/LADUMA_data/WORK

python3 -W "ignore" pred.py 1> ./outputs/out1 2> ./outputs/out2 &

exit 0
