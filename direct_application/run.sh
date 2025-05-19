#!/bin/bash

mkdir -p ./fwd_res
mkdir -p ./outputs
mkdir -p ./pred_subfiles

python3 pred.py 1> ./outputs/out1 2> ./outputs/out2 &

exit 0
