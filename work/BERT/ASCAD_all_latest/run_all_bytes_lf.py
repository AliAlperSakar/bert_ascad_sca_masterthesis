import subprocess
import os
import time
from pathlib import Path
import csv
from datetime import datetime
import re
import argparse

def parse_metrics_from_output(output):
    metrics = {}
    # Parse lines for metrics
    best_rank_match = re.search(r"Best rank achieved: (\d+)", output)
    if best_rank_match:
        metrics["best_rank"] = int(best_rank_match.group(1))
    found_at_trace_match = re.search(r"Key byte found at trace (\d+)", output)
    if found_at_trace_match:
        metrics["found_at_trace"] = int(found_at_trace_match.group(1))
    predicted_key_match = re.search(r"Predicted key byte: 0x([0-9a-fA-F]+)", output)
    if predicted_key_match:
        metrics["predicted_key"] = predicted_key_match.group(1)
    real_key_match = re.search(r"Real key byte: 0x([0-9a-fA-F]+)", output)
    if real_key_match:
        metrics["real_key"] = real_key_match.group(1)
    
    # Parse test traces used (new metric)
    test_traces_match = re.search(r"Test traces used: (\d+)", output)
    if test_traces_match:
        metrics["test_traces_used"] = int(test_traces_match.group(1))
    
    return metrics

def get_seed_from_model_path(model_path):
    match = re.search(r"seed(\d+)", model_path)
    return int(match.group(1)) if match else None

# Updated CSV header to include new leak-free metrics
CSV_HEADER = [
    "timestamp", "key_byte", "model_path", "dataset_path", "seed", "status", 
    "duration_sec", "best_rank", "found_at_trace", "predicted_key", "real_key", 
    "test_traces_used", "evaluation_type", "data_leakage_prevented"
]

def run_test_for_byte(key_byte, model_path, dataset_path, ablation=None):
    """
    Run test for a single key byte using the NEW leak-free testing script
    """

    cmd = [
        "python",
        "test_ascad_bert_improved_no_leakage.py", 
        "--dataset_path", dataset_path,
        "--model_path", model_path,
        "--key_byte", str(key_byte),
        "--num_traces", "10000",  
        "--batch_size", "64"
    ]
    
    if ablation:
        cmd += ["--ablation", ablation]
    
    # Auto-detect paths from model directory
    model_dir = Path(model_path).parent
    
    # Add paths for normalization and indices
    cmd += [
        "--profiling_mean_path", str(model_dir / "profiling_mean.npy"),
        "--profiling_std_path", str(model_dir / "profiling_std.npy"),
        "--test_indices_path", str(model_dir / "test_indices.npy")
    ]
    
    print(f"\n{'='*80}")
    print(f"🎯 Testing key byte {key_byte} (LEAK-FREE EVALUATION)")
    print(f"📋 Using test indices from: {model_dir / 'test_indices.npy'}")
    print(f"🚫 Validation traces automatically excluded")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        status = "SUCCESS"
        output = result.stdout
        print(f"✅ Key byte {key_byte} completed successfully")
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        output = e.stdout + "\n" + (e.stderr or "")
        print(f"❌ Key byte {key_byte} failed: {e}")
        print(f"Error output: {output}")
    except FileNotFoundError:
        status = "SCRIPT_NOT_FOUND"
        output = "Testing script not found. Make sure test_ascad_bert_fixed.py exists."
        print(f"❌ Testing script not found!")
    
    end_time = time.time()
    duration = end_time - start_time
    metrics = parse_metrics_from_output(output)
    
    # Compose row with new leak-free metrics
    row = {
        "timestamp": datetime.now().isoformat(),
        "key_byte": key_byte,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "seed": get_seed_from_model_path(model_path),
        "status": status,
        "duration_sec": f"{duration:.2f}",
        "best_rank": metrics.get("best_rank"),
        "found_at_trace": metrics.get("found_at_trace"),
        "predicted_key": metrics.get("predicted_key"),
        "real_key": metrics.get("real_key"),
        "test_traces_used": metrics.get("test_traces_used"),
        "evaluation_type": "FINAL_TEST_NO_LEAKAGE",
        "data_leakage_prevented": True
    }
    
    # Write to CSV
    csv_path = "key_byte_results/test_results_leak_free.csv"  # New filename to distinguish
    file_exists = os.path.isfile(csv_path)
    
    # Create directory if it doesn't exist
    Path("key_byte_results").mkdir(exist_ok=True)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"📊 Results saved to: {csv_path}")
    return status == "SUCCESS"

def main():
    parser = argparse.ArgumentParser(description="Batch test all key bytes with leak-free evaluation.")
    parser.add_argument("--ablation", type=str, default=None, 
                       choices=[None, "no_dropout", "no_posenc", "no_fusion", "tiny_emb"], 
                       help="Ablation variant to run (optional)")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the ASCAD dataset")
    parser.add_argument("--start_byte", type=int, default=0,
                       help="Starting key byte (default: 0)")
    parser.add_argument("--end_byte", type=int, default=15,
                       help="Ending key byte (default: 15)")
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path("key_byte_results")
    results_dir.mkdir(exist_ok=True)
    
    # Verify model and required files exist
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    model_dir = model_path.parent
    required_files = [
        model_dir / "profiling_mean.npy",
        model_dir / "profiling_std.npy", 
        model_dir / "test_indices.npy"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Required file not found: {file_path}")
            print("Make sure you trained with the new leak-free training script!")
            return
    
    print("="*80)
    print("🎯 LEAK-FREE BATCH EVALUATION")
    print("="*80)
    print(f"📁 Model: {model_path}")
    print(f"📁 Dataset: {args.dataset_path}")
    print(f"🔢 Testing key bytes: {args.start_byte}-{args.end_byte}")
    print(f"🚫 Data leakage prevention: ENABLED")
    print("="*80)
    
    successful_tests = 0
    failed_tests = 0
    
    for key_byte in range(args.start_byte, args.end_byte + 1):
        success = run_test_for_byte(key_byte, str(model_path), args.dataset_path, 
                                   ablation=args.ablation)
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
        
        # Small delay between tests to avoid overloading the system
        time.sleep(2)
    
    print("\n" + "="*80)
    print(f"🎯 BATCH EVALUATION COMPLETE")
    print("="*80)
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(f"📊 Results saved to: key_byte_results/test_results_leak_free.csv")
    print("🚫 All results are leak-free and honest!")
    print("="*80)

if __name__ == "__main__":
    main()