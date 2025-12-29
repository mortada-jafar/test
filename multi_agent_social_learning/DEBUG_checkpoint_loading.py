# Add this cell BEFORE running training to diagnose the issue

from pathlib import Path
import os

print("="*60)
print("CHECKPOINT LOADING DIAGNOSTICS")
print("="*60)

# Check 1: List all available datasets in Kaggle
print("\n1. Available Kaggle input directories:")
kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    for item in kaggle_input.iterdir():
        print(f"   - {item}")
else:
    print("   ⚠ /kaggle/input does not exist!")

# Check 2: Check your specific dataset path
checkpoint_dir = Path("/kaggle/input/multi-agent-model/models")
print(f"\n2. Checking path: {checkpoint_dir}")
print(f"   Exists: {checkpoint_dir.exists()}")

# Check 3: Try without the /models subdirectory
checkpoint_dir_alt = Path("/kaggle/input/multi-agent-model")
print(f"\n3. Checking alternate path: {checkpoint_dir_alt}")
print(f"   Exists: {checkpoint_dir_alt.exists()}")
if checkpoint_dir_alt.exists():
    print("   Contents:")
    for item in checkpoint_dir_alt.iterdir():
        print(f"      - {item}")

# Check 4: Look for agent files
print("\n4. Looking for agent checkpoint files:")
for possible_dir in [checkpoint_dir, checkpoint_dir_alt]:
    if possible_dir.exists():
        for i in range(3):
            agent_file = possible_dir / f"agent_{i}.pt"
            print(f"   {agent_file}: {'✓ FOUND' if agent_file.exists() else '✗ NOT FOUND'}")

print("\n" + "="*60)
print("RECOMMENDED ACTION:")
print("="*60)

# Provide recommendation
if checkpoint_dir.exists():
    print("✓ Checkpoint directory found!")
    print("→ Check if agent_0.pt, agent_1.pt, agent_2.pt files exist")
elif checkpoint_dir_alt.exists():
    print("⚠ Dataset found but at different path!")
    print(f"→ Change checkpoint_dir to: Path('{checkpoint_dir_alt}')")
else:
    print("✗ Dataset not found!")
    print("→ Did you add the dataset to this notebook?")
    print("→ Steps:")
    print("   1. Click '+ Add Data' in right sidebar")
    print("   2. Search for your dataset name")
    print("   3. Click 'Add'")
    print("   4. Re-run this diagnostic cell")

print("="*60)
