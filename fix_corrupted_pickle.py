#!/usr/bin/env python
"""
Fix corrupted pickle file by removing dynamic method references
"""

import pickle
import sys

PICKLE_FILE = 'pklfiles/20251030_US-Blo_ICB20TRCNPRDCTCBC.pkl'

print("Fixing corrupted pickle file...")

# Custom unpickler that ignores missing attributes
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Allow all classes
        return super().find_class(module, name)

# Try to load with a workaround
try:
    with open(PICKLE_FILE, 'rb') as f:
        # Load raw bytes
        data = f.read()

    # Create a dummy function to satisfy the pickle
    import types
    def dummy_func(self, *args, **kwargs):
        pass

    # Patch the module temporarily
    import model_ELM.main
    if not hasattr(model_ELM.main.ELMcase, 'run_surrogate_reduced'):
        model_ELM.main.ELMcase.run_surrogate_reduced = dummy_func
        model_ELM.main.ELMcase.run_surrogate_original = dummy_func

    # Now try to load
    mycase = pickle.loads(data)

    # Remove the problematic attributes
    if hasattr(mycase, 'run_surrogate_reduced'):
        delattr(mycase, 'run_surrogate_reduced')
    if hasattr(mycase, 'run_surrogate_original'):
        delattr(mycase, 'run_surrogate_original')

    # Also reset parameter lists to original v3 values
    parm_file = 'inputdata/PTTAM/US-Blo_parm_list_tam_v3'
    print(f"Resetting to original parameter file: {parm_file}")
    mycase.read_parm_list(parm_file)

    # Save the fixed version
    mycase.create_pkl(outdir='pklfiles/')
    print(f"✓ Fixed pickle saved to: {PICKLE_FILE}")
    print(f"✓ Case now has {mycase.nparms_ensemble} parameters")

except Exception as e:
    print(f"✗ Error: {e}")
    print(f"\nTrying alternative: restore from Oct 29 backup...")
    import shutil
    backup_file = 'pklfiles/20251029_US-Blo_ICB20TRCNPRDCTCBC.pkl'
    target_file = PICKLE_FILE

    try:
        # Make a backup of corrupted file
        shutil.copy(target_file, target_file + '.corrupted')
        print(f"  Backed up corrupted file to: {target_file}.corrupted")

        # Copy Oct 29 version
        shutil.copy(backup_file, target_file)
        print(f"  ✓ Restored from: {backup_file}")

        # Load and update parameter ranges
        with open(target_file, 'rb') as f:
            mycase = pickle.load(f)

        # Update to v3 parameter ranges
        parm_file = 'inputdata/PTTAM/US-Blo_parm_list_tam_v3'
        print(f"  Updating to parameter file: {parm_file}")
        mycase.read_parm_list(parm_file)
        mycase.create_pkl(outdir='pklfiles/')
        print(f"  ✓ Updated case saved with {mycase.nparms_ensemble} parameters")

    except Exception as e2:
        print(f"✗ Restore also failed: {e2}")
        sys.exit(1)

print("\n" + "="*70)
print("Pickle file is now fixed and ready for reduced MCMC setup")
print("="*70)
