#!/bin/bash
# Phase A.3: Batched Parameter Sweep Launch
# Run 6 trainings at a time to avoid memory overload

BATCH=$1  # 1, 2, or 3

if [ -z "$BATCH" ]; then
    echo "Usage: ./launch_batch.sh <batch_number>"
    echo "  batch 1: β=0.10"
    echo "  batch 2: β=0.17"
    echo "  batch 3: β=0.25"
    exit 1
fi

ETAS=(0.005 0.01 0.02)
LAMBDAS=(0.0 1.0)
STEPS=50000

case $BATCH in
    1)
        BETA=0.10
        ;;
    2)
        BETA=0.17
        ;;
    3)
        BETA=0.25
        ;;
    *)
        echo "Invalid batch number. Use 1, 2, or 3."
        exit 1
        ;;
esac

echo "========================================="
echo "Phase A.3: Batch $BATCH Launch"
echo "========================================="
echo "β = $BETA"
echo "η ∈ {0.005, 0.01, 0.02}"
echo "λ_H ∈ {0.0, 1.0}"
echo "Total: 6 trainings × ${STEPS} steps"
echo "========================================="
echo

count=0

for eta in "${ETAS[@]}"; do
    for lambda_h in "${LAMBDAS[@]}"; do
        count=$((count + 1))
        
        config_name="β=${BETA}_η=${eta}_λ=${lambda_h}"
        echo "[$count/6] Launching: $config_name"
        
        # Launch in background
        nohup python3 train_parameter_sweep_v2.py \
            --beta "$BETA" \
            --eta "$eta" \
            --lambda_h "$lambda_h" \
            --steps "$STEPS" \
            > "sweep_logs/sweep_beta${BETA}_eta${eta}_lambda${lambda_h}.log" 2>&1 &
        
        pid=$!
        echo "  PID: $pid"
        echo "$pid" > "sweep_logs/sweep_beta${BETA}_eta${eta}_lambda${lambda_h}.pid"
        
        sleep 2
    done
done

echo
echo "========================================="
echo "Batch $BATCH: 6 trainings launched!"
echo "========================================="
echo "Monitor: tail -f sweep_logs/sweep_beta${BETA}*.log"
echo "Check: ps aux | grep train_parameter_sweep"
echo
echo "When complete, launch next batch:"
echo "  ./launch_batch.sh $((BATCH + 1))"
echo "========================================="
