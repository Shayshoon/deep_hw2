#!/bin/bash

# Make sure py-sbatch.sh is executable
chmod +x py-sbatch.sh

# --- Usage Instructions ---
# You can now pass any flag to this script, and it will be forwarded to py-sbatch.
#
# To reproduce your original hardcoded configuration, run:
# ./submit_experiments.sh -P 4 -H 75 -M cnn --lr 0.0001 --reg 0.02 --epochs 75 --bs-train 64 --batches 50
#
# If you run it without flags:
# ./submit_experiments.sh
# Then ONLY -n, -K, and -L are passed.

# --- Experiment 1 Loop ---
for K in 32; do
    for L in 2 4 8 16; do

        # 1. Construct the specific run name per instructions
        RUN_NAME="exp1_1_L${L}_K${K}"

        echo "Submitting: $RUN_NAME (K=$K, L=$L)"

        # 2. Call your existing wrapper script
        # -n, -K, and -L are set automatically.
        # "$@" expands to all arguments passed to this shell script.
        ./py-sbatch.sh -m hw2.experiments run-exp \
            -n "$RUN_NAME" \
            -K "$K" \
            -L "$L" \
            "$@"

    done
done
