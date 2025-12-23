#!/bin/bash
# =========================================================
# Experiment Configuration
# Edit these values to control the experiment parameters.
# If you leave a value empty (""), the python script will use its internal default.
# =========================================================

# --- Mandatory Architecture Params ---
# Note: K (Filters) and L (Layers) are handled automatically by the loop logic
POOL_EVERY="4"          # -P
HIDDEN_DIMS="100"       # -H
MODEL_TYPE="cnn"        # -M

# --- Training Hyperparameters ---
BATCH_SIZE_TRAIN="128"   # --bs-train
BATCH_SIZE_TEST=""      # --bs-test (Empty = Auto)
EPOCHS="75"             # --epochs
EARLY_STOPPING="5"      # --early-stopping
LEARNING_RATE="0.0005"  # --lr
REGULARIZATION="0.002"  # --reg
BATCHES_PER_EPOCH="100"  # --batches (Set to "" for full epoch)
SEED=""                 # --seed
CHECKPOINTS=""          # --checkpoints

# --- System ---
OUT_DIR="./results"     # -o
