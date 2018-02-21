#!/bin/bash

function show_help {
    echo "---"
    echo "Usage:"
    echo "      ./run_train.sh [GPU_ID]"
    echo ""
    echo "Arguments:"
    echo "  -h or --help    Prints this help"
    echo "  GPU_ID          [optional; default value: 0] positive integer defining the GPU to use"
    echo ""
    echo ""
    echo "The parameters following the first one will be ignored."
    echo "---"
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

if [ $# -eq 0 ]; then
    GPU_ID=""
elif [ "$1" -lt 0 ]; then
    echo "Error: Invalid GPU_ID value"
    show_help
    exit -3
else
    GPU_ID="--gpu_id $1"
fi

train_args="--model alexnet --dataset_dir data/bin_data --epochs 3 --batch_size 16 --verbose --use_cuda $GPU_ID"

echo "--- TRAIN ---"
python3 train.py $train_args
