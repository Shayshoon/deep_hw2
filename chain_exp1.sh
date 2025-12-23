#!/bin/bash

# =========================================================
# Daisy-Chain Scheduler (Configurable)
# Usage: ./chain_experiment.sh [STEP_INDEX]
# =========================================================

# 1. Load the Configuration
# -------------------------
CONFIG_FILE="./experiment_config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi
source $CONFIG_FILE

# 2. Step Logic (K and L loop)
# ----------------------------
IDX=${1:-0}

# Define the experiment chain logic (same as before)
if [ $IDX -ge 8 ]; then
    echo "All 8 experiments in the chain are complete!"
    exit 0
fi

if [ $IDX -lt 4 ]; then
    K=32
    L_IDX=$IDX
else
    K=64
    L_IDX=$((IDX - 4))
fi

L_OPTIONS=(2 4 8 16)
L=${L_OPTIONS[$L_IDX]}
RUN_NAME="exp1_1_L${L}_K${K}"

echo ">>> Processing Chain Link #$IDX: $RUN_NAME (K=$K, L=$L)"

# 3. Construct the Python Command String
# --------------------------------------
# We use bash syntax ${VAR:+...} which means:
# "If VAR is set, print '...'. If VAR is empty, print nothing."
# This lets the Python script use its defaults if params are missing.

PY_ARGS="-n $RUN_NAME -K $K -L $L -P $POOL_EVERY -H $HIDDEN_DIMS -M $MODEL_TYPE"

# Optional flags (only added if set in config)
PY_ARGS+=${OUT_DIR:+ --out-dir $OUT_DIR}
PY_ARGS+=${BATCH_SIZE_TRAIN:+ --bs-train $BATCH_SIZE_TRAIN}
PY_ARGS+=${BATCH_SIZE_TEST:+ --bs-test $BATCH_SIZE_TEST}
PY_ARGS+=${BATCHES_PER_EPOCH:+ --batches $BATCHES_PER_EPOCH}
PY_ARGS+=${EPOCHS:+ --epochs $EPOCHS}
PY_ARGS+=${EARLY_STOPPING:+ --early-stopping $EARLY_STOPPING}
PY_ARGS+=${LEARNING_RATE:+ --lr $LEARNING_RATE}
PY_ARGS+=${REGULARIZATION:+ --reg $REGULARIZATION}
PY_ARGS+=${SEED:+ --seed $SEED}
PY_ARGS+=${CHECKPOINTS:+ --checkpoints $CHECKPOINTS}

# 4. Slurm Submission
# -------------------
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
MAIL_USER="shay-lavi@technion.ac.il"
MAIL_TYPE=FAIL 
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw2

sbatch \
    -N $NUM_NODES \
    -c $NUM_CORES \
    --nodelist=lambda2 \
    --gres=gpu:$NUM_GPUS \
    --job-name $RUN_NAME \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    -o "slurm-${RUN_NAME}-%j.out" \
<<EOF
#!/bin/bash
echo "*** CHAIN JOB #$IDX: '$RUN_NAME' STARTING ***"

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run the command with the constructed arguments
echo "Running: python -m hw2.experiments run-exp $PY_ARGS"
python -m hw2.experiments run-exp $PY_ARGS

EXIT_CODE=\$?

if [ \$EXIT_CODE -eq 0 ]; then
    echo "Experiment successful."
    NEXT_IDX=\$(( $IDX + 1 ))
    echo "Chaining next job: Index \$NEXT_IDX..."
    
    # Recursively call this script
    /bin/bash \$SLURM_SUBMIT_DIR/chain_exp1.sh \$NEXT_IDX
else
    echo "Experiment failed with exit code \$EXIT_CODE!"
    exit \$EXIT_CODE
fi
EOF
