#!/bin/bash
# Phase 3 Verification Script
# Runs all tests and checks to confirm implementation is complete

set -e  # Exit on error

cd /Users/yz/.openclaw/workspace-xiaoa/ptm-rl
export PYTHONPATH=$PWD:$PYTHONPATH

echo "======================================================================="
echo "Phase 3 Verification Script"
echo "======================================================================="
echo ""

echo "1. Checking file structure..."
echo "   - Environment implementation..."
[ -f "src/pytokmhd/rl/mhd_env_v1_4.py" ] && echo "     ✅ mhd_env_v1_4.py" || exit 1

echo "   - Tests..."
[ -f "tests/rl/test_mhd_env_v1_4.py" ] && echo "     ✅ test_mhd_env_v1_4.py" || exit 1

echo "   - Documentation..."
[ -f "docs/phase3_implementation_report.md" ] && echo "     ✅ Implementation report" || exit 1
[ -f "README_v1_4.md" ] && echo "     ✅ Quick start guide" || exit 1
[ -f "PHASE3_COMPLETION.md" ] && echo "     ✅ Completion summary" || exit 1

echo ""
echo "2. Running unit tests..."
python3 -m pytest tests/rl/test_mhd_env_v1_4.py -v --tb=short

echo ""
echo "3. Testing import..."
python3 -c "from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D; print('   ✅ Import successful')"

echo ""
echo "4. Quick functionality test (5 steps, zero action)..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/yz/.openclaw/workspace-xiaoa/ptm-rl')
import numpy as np
from src.pytokmhd.rl.mhd_env_v1_4 import MHDEnv3D

env = MHDEnv3D(grid_size=(16, 32, 16), max_steps=5)
obs, info = env.reset(seed=42)
print(f"   Initial E₀ = {info['E0']:.3e}")

for step in range(5):
    action = np.zeros(5)  # Zero action
    obs, reward, terminated, truncated, info = env.step(action)

print(f"   Final drift = {info['energy_drift']:.3e}")
print("   ✅ Functionality test passed")
EOF

echo ""
echo "======================================================================="
echo "Phase 3 Verification Complete!"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  ✅ All files present"
echo "  ✅ All tests passing"
echo "  ✅ Environment functional"
echo ""
echo "Status: READY FOR PHASE 4 (RL Training)"
echo ""
