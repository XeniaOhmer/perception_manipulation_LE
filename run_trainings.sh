#!/bin/bash

let GPU=1

orig_pcount=$(ps -ef | grep -v grep | grep mnist_training.py | wc -l)
limit=4

for PARAMS_PATH in param_files/* ; do
    while [ $((process_count)) -ge $((limit)) ]
    do
        sleep 30
        process_count=$(ps -ef | grep -v grep | grep mnist_training.py | wc -l)
        process_count=$((process_count))-$((orig_pcount))
    done
    echo "[BASH INFO] running training for exp. $PARAMS_PATH"
    python mnist_training.py -g $GPU -p $PARAMS_PATH -b 1 &
    echo "[BASH INFO] Training experiment complete"
    let GPU=$GPU-1
    if [ $GPU -lt 0 ]; then
        let GPU=-$GPU
    fi
    process_count=$(ps -ef | grep -v grep | grep mnist_training.py | wc -l)
    process_count=$((process_count))-$((orig_pcount))
done
