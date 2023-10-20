#!/bin/bash

if [ -z $1 ];
then
    echo "input directory not found";
    exit;
else
    echo "read input from '$1'";
fi
file=$1

set -x
CUDA_VISIBLE_DEVICES=0 python3 -W ignore main.py --embedding-dim=768 \
    --batch_size=4444 --num_epochs=5 --lr=0.01 \
