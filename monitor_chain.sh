#!/bin/bash

# =========================================================
# Slurm Job Auto-Follower
# Usage: ./monitor_chain.sh [USERNAME]
# Description: Automatically detects running jobs and tails 
#              their output files. Handles daisy-chained jobs.
# =========================================================

TARGET_USER=${1:-$USER}

echo ">>> Starting Auto-Monitor for user: $TARGET_USER"
echo ">>> Press Ctrl+C to stop."

while true; do
    # 1. Find the currently RUNNING job ID (State = 'R')
    #    -h: No header, -t R: Running only, -o %i: Print JobID only
    JOB_ID=$(squeue -u $TARGET_USER -t R -h -o %i | head -n 1)

    # 2. If no job is running, wait and retry
    if [ -z "$JOB_ID" ]; then
        echo -ne ">>> No running jobs detected. Waiting... $(date +%H:%M:%S)\r"
        sleep 3
        continue
    fi

    echo ""
    echo "==========================================================="
    echo ">>> DETECTED RUNNING JOB: $JOB_ID"
    echo "==========================================================="

    # 3. Find the specific log file for this Job ID
    #    We loop briefly because Slurm takes a second to create the file.
    LOG_FILE=""
    MAX_RETRIES=10
    COUNT=0

    while [ -z "$LOG_FILE" ] && [ $COUNT -lt $MAX_RETRIES ]; do
        # Look for any file ending in "-<JOB_ID>.out"
        LOG_FILE=$(ls slurm-*${JOB_ID}.out 2>/dev/null | head -n 1)
        
        if [ -z "$LOG_FILE" ]; then
            echo "   (Waiting for log file creation...)"
            sleep 1
            ((COUNT++))
        fi
    done

    if [ -z "$LOG_FILE" ]; then
        echo ">>> Error: Log file for Job $JOB_ID not found after 10 seconds."
        echo ">>> Skipping to next check..."
        sleep 5
        continue
    fi

    echo ">>> TAILING FILE: $LOG_FILE"
    echo "-----------------------------------------------------------"

    # 4. Tail the file in the background
    tail -f "$LOG_FILE" &
    TAIL_PID=$!

    # 5. Monitor the job status; Keep tailing while job exists
    while true; do
        # Check if job is still in squeue
        if ! squeue -j "$JOB_ID" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    # 6. Job finished: Kill the tail process and loop again
    kill $TAIL_PID 2>/dev/null
    wait $TAIL_PID 2>/dev/null
    
    echo ""
    echo "-----------------------------------------------------------"
    echo ">>> Job $JOB_ID has finished (or crashed)."
    echo ">>> Searching for next job..."
    echo ""
done
