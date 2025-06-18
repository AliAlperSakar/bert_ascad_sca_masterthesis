#!/usr/bin/env python3
import subprocess
import os
import time
from pathlib import Path

def run_test_for_byte(key_byte):
    cmd = [
        "python",
        "/root/alper/jupyter_transfer/downloaded_data/work/BERT/ASCAD_all_latest/test_ascad_bert_improved.py",
        "--dataset_path", "/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable-desync50.h5",
        "--model_path", "/root/alper/jupyter_transfer/downloaded_data/work/BERT/ASCAD_all_latest/bert_sca_ascad-variable-desync50_seed222_e4_lr1e-04_s50000.pth",
        "--key_byte", str(key_byte)
    ]
    
    print(f"\n{'='*80}")
    print(f"Testing key byte {key_byte}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running test for key byte {key_byte}: {e}")
        return False
    
    return True

def main():
    # Create a results directory if it doesn't exist
    results_dir = Path("key_byte_results")
    results_dir.mkdir(exist_ok=True)
    
    # Test all 16 bytes
    for key_byte in range(16):
        start_time = time.time()
        success = run_test_for_byte(key_byte)
        end_time = time.time()
        
        # Log the results
        with open(results_dir / "test_results.txt", "a") as f:
            status = "SUCCESS" if success else "FAILED"
            duration = end_time - start_time
            f.write(f"Key byte {key_byte:02d}: {status} (Duration: {duration:.2f} seconds)\n")
        
        # Add a small delay between runs to ensure clean separation
        time.sleep(2)

if __name__ == "__main__":
    main() 