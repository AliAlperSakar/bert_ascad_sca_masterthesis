import numpy as np
import torch
import h5py
import wandb
import argparse
import os
from train_ascad_bert import BERT_SCA_Model, BertModel, tokenizer
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Define DEVICE constant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AES Sbox
AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def load_test_data(dataset_path, profiling_mean_path='profiling_mean.npy', profiling_std_path='profiling_std.npy'):
    print(f"Loading dataset: {dataset_path}")
    with h5py.File(dataset_path, 'r') as in_file:
        X_test = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        plaintexts = np.array(in_file['Attack_traces/metadata'][:]['plaintext'])
        
        # Load profiling mean and std
        keys = np.array(in_file['Attack_traces/metadata'][:]['key'])
        if not os.path.exists(profiling_mean_path) or not os.path.exists(profiling_std_path):
            raise FileNotFoundError(f"Profiling mean/std files not found. Make sure you run training first and save them. Expected: {profiling_mean_path}, {profiling_std_path}")
        
        # Normalize traces
        mean = np.load(profiling_mean_path)
        std = np.load(profiling_std_path)

        std[std == 0] = 1e-8 

        X_test = (X_test - mean) / std
        
    return X_test, plaintexts, keys

def compute_rank(predictions, plaintext, key, byte, index):
    key_bytes_proba = np.zeros(256)
    plaintext_byte = plaintext[index][byte]
    real_key = key[index][byte]
    
    for key_byte in range(256):
        sbox_output = AES_Sbox[plaintext_byte ^ key_byte]
        key_bytes_proba[key_byte] = predictions[sbox_output]
    
    sorted_proba = np.argsort(key_bytes_proba)[::-1]
    key_rank = np.where(sorted_proba == real_key)[0][0]
    predicted_key = sorted_proba[0]
    
    return key_rank, predicted_key, real_key

def predict_batch(model, traces, plaintexts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    plaintext_strs = [' '.join([f'{b:02x}' for b in p]) for p in plaintexts]
    encodings = tokenizer(plaintext_strs, padding=True, truncation=True, return_tensors="pt")
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    traces = torch.tensor(traces, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, traces)
        predictions = torch.softmax(outputs, dim=1).cpu().numpy()
    
    return predictions

def evaluate_key_rank(model, X_test, plaintexts, keys, key_byte, num_traces, batch_size=100):
    print(f"Evaluating key byte: {key_byte}, Processing {num_traces} traces...")
    ranks = np.zeros((num_traces, 2))
    key_found = False
    key_info = {"key_found": False, "found_at_trace": None, "best_rank": 256}
    best_rank_so_far = 256  # Track best rank achieved
    
    for batch_start in range(0, num_traces, batch_size):
        batch_end = min(batch_start + batch_size, num_traces)
        batch_traces = X_test[batch_start:batch_end]
        batch_plaintexts = plaintexts[batch_start:batch_end]
        
        predictions = predict_batch(model, batch_traces, batch_plaintexts)
        
        for i in range(len(batch_traces)):
            idx = batch_start + i
            rank, predicted_key, real_key = compute_rank(predictions[i], plaintexts, keys, key_byte, idx)
            best_rank_so_far = min(best_rank_so_far, rank)  # Update best rank
            ranks[idx][0] = idx
            ranks[idx][1] = best_rank_so_far  # Store best rank achieved so far
            
            if rank == 0 and not key_found:
                key_found = True
                key_info.update({
                    "key_found": True,
                    "found_at_trace": idx,
                    "best_rank": 0,
                    "predicted_key": predicted_key,
                    "real_key": real_key
                })
                print(f"Key found at trace {idx} with rank 0")
    
    if not key_found:
        print("Key not found within the given traces.")
        key_info["best_rank"] = best_rank_so_far
    
    return ranks, key_info

def plot_key_rank_evolution(ranks):
    """
    Creates a ranking evolution plot using matplotlib with a thicker line,
    and logs it to WandB as an image.
    
    Parameters:
        ranks (np.ndarray): A 2D array where column 0 is the trace indices and column 1
                            is the key rank evolution.
    """
    import matplotlib.pyplot as plt

    # Extract trace indices and ranks
    trace_indices = ranks[:, 0]
    key_ranks = ranks[:, 1]
    
    # Create the plot with custom styling (e.g., thicker line)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(trace_indices, key_ranks, label="Key Rank Evolution", color='blue', linewidth=3)  # linewidth=3 makes the line thicker
    ax.set_xlabel("Trace Index")
    ax.set_ylabel("Key Rank")
    ax.set_title("Key Rank Evolution Over Number of Traces")
    ax.grid(True)
    ax.legend()
    
    # Log the figure to WandB
    import wandb
    wandb.log({"Key Rank Evolution Plot": wandb.Image(fig)})
    
    # Optionally, display or save the figure as needed
    plt.show()
    # Alternatively, if you don't need to display it:
    # plt.close(fig)


# 🚀 New: WandB Initialization Block with Auto-Naming
def wandb_initialize(args):
    dataset_name = Path(args.dataset_path).stem  # e.g., "ascad-variable-desync50"
    run_name = f"Byte-{args.key_byte:02d}"        # e.g., Byte-00, Byte-01, ...
    project_name = f"ascad-bert-key-eval-{dataset_name}"  # Unique project per dataset variant

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "num_traces": args.num_traces,
            "batch_size": args.batch_size,
            "key_byte": args.key_byte
        }
    )
    return project_name, run_name
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT model on ASCAD dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset (.h5)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--num_traces", type=int, default=10000, help="Number of traces to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--key_byte", type=int, default=0, help="Key byte index to evaluate")
    parser.add_argument("--output_csv", type=str, default="rank_results.csv", help="Output CSV file for rank results")
    parser.add_argument("--output_plot", type=str, default="key_rank_evolution.png", help="Output file for rank evolution plot")
    args = parser.parse_args()


    wandb_initialize(args)


    print("Loading model...")
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SCA_Model(bert_model, trace_length=X_test.shape[1])
    # Load the state dict with proper error handling
    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # If the saved file is a checkpoint dictionary
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If the saved file is just the state dict
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    model = model.to(DEVICE)
    
    X_test, plaintexts, keys = load_test_data(args.dataset_path, 'profiling_mean.npy', 'profiling_std.npy')
    
    ranks, key_info = evaluate_key_rank(
        model, X_test, plaintexts, keys, args.key_byte, args.num_traces, args.batch_size
    )
    
    # Process ranks for smoother evolution
    processed_ranks = []
    current_rank = 256  # Start with worst possible rank
    
    for i in range(len(ranks)):
        trace_idx = int(ranks[i][0])
        rank_value = int(ranks[i][1])
        # Update current rank only if it improves
        current_rank = min(current_rank, rank_value)
        processed_ranks.append([trace_idx, current_rank])
    
    # Create the ranking plot data
    table = wandb.Table(
        data=processed_ranks,
        columns=["Number of Traces", "Key Rank"]
    )
    
    # Log the rank evolution plot
    wandb.log({
        "Key Rank Evolution": wandb.plot.line(
            table,
            "Number of Traces",   # ← x-axis label
            "Key Rank",           # ← y-axis label
            title="Key Rank Evolution over Number of Traces"
        )
    })
    
    # Log statistics and key information
    wandb.log({
        "final_rank": float(processed_ranks[-1][1]),
        "min_rank": float(min(r[1] for r in processed_ranks)),
        "max_rank": float(max(r[1] for r in processed_ranks)),
        "avg_rank": float(np.mean([r[1] for r in processed_ranks])),
        "key_found": key_info["key_found"],
        "found_at_trace": key_info["found_at_trace"],
        "best_rank": key_info["best_rank"]
    })
    
    # Print final results
    print("\nFinal Results:")
    print(f"Best rank achieved: {key_info['best_rank']}")   
    print(f"Final rank: {processed_ranks[-1][1]}")
    if key_info["key_found"]:
        print(f"Key found at trace {key_info['found_at_trace']}")
        print(f"Predicted key: 0x{key_info['predicted_key']:02x}")
        print(f"Real key: 0x{key_info['real_key']:02x}")
    
    wandb.finish()
        # Save evaluation config
    eval_config = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "key_byte": args.key_byte,
        "num_traces": args.num_traces,
        "batch_size": args.batch_size,
        "embedding": "hex",  # adjust as needed
        "backbone": "bert-base-uncased",
        "trace_net": "fc",
        "fusion": "concat"
    }

    eval_config_path = os.path.splitext(args.model_path)[0] + "_eval_config.yaml"
    with open(eval_config_path, 'w') as f:
        yaml.dump(eval_config, f)
    print(f"✅ Evaluation config saved to {eval_config_path}")

if __name__ == "__main__":
    main()