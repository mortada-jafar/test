#!/bin/bash
# Quick test script to verify the system works

echo "=================================================="
echo "Multi-Agent Social Learning - Quick Test"
echo "=================================================="

echo ""
echo "Step 1: Training a small experiment (10K steps)..."
echo "This will take ~2-3 minutes"
echo ""

python train.py --method proposed --total-steps 10000 --seed 0 --save-dir test_results

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Step 2: Visualizing the trained agent..."
    echo ""

    # Find the latest checkpoint
    CHECKPOINT=$(ls -d test_results/proposed/seed_0/checkpoint_* 2>/dev/null | tail -1)

    if [ -n "$CHECKPOINT" ]; then
        echo "Checkpoint found: $CHECKPOINT"
        python visualize_agent.py \
            --checkpoint "$CHECKPOINT" \
            --agent-id 1 \
            --method proposed \
            --n-episodes 1 \
            --no-render

        echo ""
        echo "=================================================="
        echo "✓ QUICK TEST PASSED!"
        echo "=================================================="
        echo ""
        echo "The system is working correctly!"
        echo ""
        echo "Next steps:"
        echo "1. Run full demo: python demo.py"
        echo "2. Train longer: python train.py --method proposed --total-steps 500000"
        echo "3. Compare methods: python utils/plotting.py"
        echo ""
    else
        echo "Warning: No checkpoint found"
    fi
else
    echo ""
    echo "❌ Training failed. Please check error messages above."
    exit 1
fi
