#!/bin/bash
# Phase A.3 DOWNGRADED: Baseline-only Parameter Sweep
# Test environment generalization across (β, η) without Hamiltonian comparison

BETAS=(0.10 0.17 0.25)
ETAS=(0.005 0.01 0.02)
STEPS=50000

echo "========================================="
echo "Phase A.3 (Downgraded): Baseline Sweep"
echo "========================================="
echo "Grid: β ∈ {0.10, 0.17, 0.25}"
echo "      η ∈ {0.005, 0.01, 0.02}"
echo "Config: Baseline PPO only (λ_H=0.0)"
echo "Total: 9 trainings × ${STEPS} steps"
echo "========================================="
echo

count=0

for beta in "${BETAS[@]}"; do
    for eta in "${ETAS[@]}"; do
        count=$((count + 1))
        
        config_name="β=${beta}_η=${eta}"
        echo "[$count/9] Launching: $config_name"
        
        # Launch baseline only
        nohup python3 train_parameter_sweep_v2.py \
            --beta "$beta" \
            --eta "$eta" \
            --lambda_h 0.0 \
            --steps "$STEPS" \
            > "sweep_logs/baseline_beta${beta}_eta${eta}.log" 2>&1 &
        
        pid=$!
        echo "  PID: $pid"
        echo "$pid" > "sweep_logs/baseline_beta${beta}_eta${eta}.pid"
        
        sleep 2
    done
done

echo
echo "========================================="
echo "9 baseline trainings launched!"
echo "========================================="
echo "Monitor: tail -f sweep_logs/baseline_*.log"
echo
echo "Estimated completion: 1-2 days (3 batches)"
echo "========================================="
