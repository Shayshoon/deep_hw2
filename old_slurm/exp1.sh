#!/bin/bash

# Make sure py-sbatch.sh is executable
chmod +x py-sbatch.sh

# --- Experiment 1 Loop ---
# Configurations: K=32 and K=64, with L=2,4,8,16 for each.

for K in 32; do
    for L in 2 4 8 16; do
        
        # 1. Construct the specific run name per instructions
        RUN_NAME="exp1_1_L${L}_K${K}"
        
        echo "Submitting: $RUN_NAME (K=$K, L=$L)"
        
        # 2. Call your existing wrapper script
        # This passes all following arguments to 'python' inside the batch job.
        ./py-sbatch.sh -m hw2.experiments run-exp \
            -n $RUN_NAME \
            -K $K \
            -L $L \
            -P 4 \
            -H 75 \
            -M cnn \
	    --lr 0.0001 \
	    --reg 0.02 \
	    --epochs 75 \
	    --bs-train 64 \
	    --batches 50
            
    done
done
