#!/bin/bash

# =========================================================
# Daisy-Chain Scheduler (Dynamic Name & Updated Logic)
# Usage: ./your_script_name.sh [STEP_INDEX]
# =========================================================

# 1. Load the Configuration
# -------------------------
CONFIG_FILE="./config_exp.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi
source $CONFIG_FILE

# Get the current script name dynamically
SCRIPT_NAME=$(basename "$0")

# 2. Step Logic (New Configuration: L=2,4,8 with K=32,64,128)
# -----------------------------------------------------------
# 1. Parse Arguments: Experiment ID and Index
#    Usage: ./script.sh [EXP_ID] [IDX]
#    Default: Exp 1 (1.1), Index 0
#    Mapping: 1->1.1, 2->1.2, 3->1.3, 4->1.4
EXP_ID=${1:-1}
IDX=${2:-0}



# 2. Define Configuration Logic based on Experiment ID
case $EXP_ID in

    1)  # === Experiment 1.1 ===
        # K=32 fixed (L=2,4,8,16) -> Indices 0-3
        # K=64 fixed (L=2,4,8,16) -> Indices 4-7
        TOTAL_RUNS=8
        MODEL_TYPE="cnn"
        
        if [ $IDX -ge $TOTAL_RUNS ]; then
            echo "Exp 1.1 finished. Switching to Exp 1.2..."
            /bin/bash $0 2 0
            exit 0
        fi

        L_OPTIONS=(2 4 8 16)
        
        if [ $IDX -lt 4 ]; then
            K="32"
            K_TAG="32"
            L=${L_OPTIONS[$IDX]}
        else
            K="64"
            K_TAG="64"
            L=${L_OPTIONS[$((IDX - 4))]}
        fi
        ;;

    2)  # === Experiment 1.2 ===
        # L=2 (K=32,64,128) -> Indices 0-2
        # L=4 (K=32,64,128) -> Indices 3-5
        # L=8 (K=32,64,128) -> Indices 6-8
        TOTAL_RUNS=9
        MODEL_TYPE="cnn"
        
        if [ $IDX -ge $TOTAL_RUNS ]; then
            echo "Exp 1.2 finished. Switching to Exp 1.3..."
            /bin/bash $0 3 0
            exit 0
        fi

        L_OPTIONS=(2 4 8)
        K_OPTIONS=(32 64 128)

        L_IDX=$((IDX / 3))
        K_IDX=$((IDX % 3))

        L=${L_OPTIONS[$L_IDX]}
        K=${K_OPTIONS[$K_IDX]}
        K_TAG=$K
        ;;

    3)  # === Experiment 1.3 ===
        # K=[64, 128] fixed, L=2,3,4 varying
        TOTAL_RUNS=3
        MODEL_TYPE="cnn"
        
        if [ $IDX -ge $TOTAL_RUNS ]; then
            echo "Exp 1.3 finished. Switching to Exp 1.4..."
            /bin/bash $0 4 0
            exit 0
        fi

        K="64 128"      # Space-separated for Python args
        K_TAG="64-128"  # Hyphenated for Filename
        
        L_OPTIONS=(2 3 4)
        L=${L_OPTIONS[$IDX]}
        ;;

    4)  # === Experiment 1.4 ===
        # K=[32] fixed, L=8,16,32 -> Indices 0-2
        # K=[64,128,256] fixed, L=2,4,8 -> Indices 3-5
        TOTAL_RUNS=6
        MODEL_TYPE="resnet"
        
        if [ $IDX -ge $TOTAL_RUNS ]; then
            echo "All experiments (1.1 - 1.4) are complete!"
            exit 0
        fi

        if [ $IDX -lt 3 ]; then
            K="32"
            K_TAG="32"
            L_OPTIONS=(8 16 32)
            L=${L_OPTIONS[$IDX]}
        else
            K="64 128 256"
            K_TAG="64-128-256"
            L_OPTIONS=(2 4 8)
            L=${L_OPTIONS[$((IDX - 3))]}
        fi
        ;;
    5)  # === Experiment 2 ===
        # Model: YourCNN (ycn)
        # K=[32, 64, 128] fixed, L=3,6,9,12 varying
        TOTAL_RUNS=4
        MODEL_TYPE="yourcnn"  # Make sure this matches your python registration

        if [ $IDX -ge $TOTAL_RUNS ]; then
            echo "Experiment 2 is complete!"
            exit 0
        fi

        K="32 64 128"
        K_TAG="32-64-128"

        L_OPTIONS=(3 6 9 12)
        L=${L_OPTIONS[$IDX]}
        
        # === DYNAMIC P CALCULATION ===
        # Since K has 3 stages, we split L evenly among them.
        # L=3 -> P=1, L=6 -> P=2, etc.
        POOL_EVERY=$(( L / 3 ))
        ;;
    *)
        echo "Invalid Experiment ID: $EXP_ID"
        exit 1
        ;;
esac

# 3. Construct Run Name
# Note: We use K_TAG for the filename to handle lists cleanly (e.g. K64-128)
RUN_NAME="exp1_${EXP_ID}_L${L}_K${K_TAG}"

echo ">>> Processing Exp 1.$EXP_ID | Link #$IDX: $RUN_NAME (L=$L, K=[$K])"

# 3. Construct the Python Command String
# --------------------------------------
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
NEXT_IDX_VAL=$(( IDX + 1 ))


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
    
    # Recursively call this script with current EXP_ID and NEXT_IDX
    # The top logic will handle switching EXP_ID if NEXT_IDX is too high.
    echo "Chaining next job: Exp $EXP_ID, Index $NEXT_IDX_VAL..."
    /bin/bash \$SLURM_SUBMIT_DIR/$SCRIPT_NAME $EXP_ID $NEXT_IDX_VAL
else
    echo "Experiment failed with exit code \$EXIT_CODE!"
    exit \$EXIT_CODE
fi
EOF
