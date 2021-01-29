#!/bin/bash

let GPU=1

orig_pcount=$(ps -ef | grep -v grep | grep dev_training_script.py | wc -l)
limit=4

for PARAMS_PATH in param_files/* ; do
    for SF in 0.0 0.1 0.2 0.3 0.4 0.5 0.6; do
        while [ $((process_count)) -ge $((limit)) ]
        do
            sleep 30
            process_count=$(ps -ef | grep -v grep | grep dev_training_script.py | wc -l)
            process_count=$((process_count))-$((orig_pcount))
        done
        echo "[BASH INFO] running training for exp. $PARAMS_PATH"
        python dev_training_script.py -g $GPU -p $PARAMS_PATH -s $SF &
        echo "[BASH INFO] Training experiment complete"
        let GPU=$GPU-1
        if [ $GPU -lt 0 ]; then
            let GPU=-$GPU
        fi
        process_count=$(ps -ef | grep -v grep | grep dev_training_script.py | wc -l)
        process_count=$((process_count))-$((orig_pcount))
    done
done
