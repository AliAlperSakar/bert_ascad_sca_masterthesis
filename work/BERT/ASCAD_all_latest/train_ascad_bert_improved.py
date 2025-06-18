import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import h5py
import wandb
from tqdm import tqdm
import random

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set up device and tokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_ascad_data(file_path, downsample=None):
    """
    Load ASCAD dataset from the given HDF5 file.
    Expects groups 'Profiling_traces' and 'Attack_traces':
      - Pr<ofiling_traces/traces: training traces
      - Profiling_traces/labels: training labels
      - Profiling_traces/metadata: training metadata (plaintext bytes)
      - Attack_traces/traces: testing traces
      - Attack_traces/labels: testing labels
      - Attack_traces/metadata: testing metadata (plaintext bytes)
    downsample: if provided, randomly selects a subset of the profiling set.
    """
    print(f"Loading dataset: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    with h5py.File(file_path, 'r') as in_file:
        # Load profiling (training) traces and metadata
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        plaintexts_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'])
        
        # Load attack (testing) traces and metadata
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        Y_attack = np.array(in_file['Attack_traces/labels'])
        plaintexts_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'])

        # Normalize using training (profiling) statistics
        mean = np.mean(X_profiling, axis=0, keepdims=True)
        std = np.std(X_profiling, axis=0, keepdims=True)

        # Handle zero std to prevent division by zero
        std[std == 0] = 1e-8 # Add a small epsilon to zero std values
        X_profiling = (X_profiling - mean) / std
        X_attack = (X_attack - mean) / std

        # Optionally downsample the profiling set
        if downsample is not None:
            indices = np.random.choice(len(X_profiling), downsample, replace=False)
            X_profiling = X_profiling[indices]
            Y_profiling = Y_profiling[indices]
            plaintexts_profiling = plaintexts_profiling[indices]
    
    return (X_profiling, Y_profiling, plaintexts_profiling), \
           (X_attack, Y_attack, plaintexts_attack), mean, std

class ASCADDataset(Dataset):
    def __init__(self, traces, labels, plaintexts, tokenizer):
        self.traces = traces
        self.labels = labels
        self.plaintexts = plaintexts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        plaintext = self.plaintexts[idx]
        
        # Convert plaintext bytes (e.g., integers) to a hex string
        plaintext_str = ' '.join([f'{b:02x}' for b in plaintext])
        
        # Tokenize the plaintext
        encoding = self.tokenizer.encode_plus(
            plaintext_str,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'trace': torch.tensor(trace, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Custom collate function to stack individual dictionary fields into batched tensors.
    """
    collated = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'trace': torch.stack([item['trace'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }
    return collated

class BERT_SCA_Model(nn.Module):
    def __init__(self, bert_model, trace_length):
        super(BERT_SCA_Model, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        # Map trace features to 768-dim to match BERT's output size
        self.trace_fc = nn.Linear(trace_length, 768)
        # Concatenate [BERT_output, trace_features] and classify to 256 classes
        self.classifier = nn.Linear(768 * 2, 256)

    def forward(self, input_ids, attention_mask, traces):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.pooler_output  # (batch_size, 768)
        trace_features = self.trace_fc(traces)     # (batch_size, 768)
        combined = torch.cat((bert_output, trace_features), dim=1)
        x = self.dropout(combined)
        logits = self.classifier(x)
        return logits

def train_model(args):
    # Initialize WandB for experiment logging
    # Dynamically build project and run names
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    project_name = f"bert-sca-{dataset_name}"
    run_name = (
        f"{dataset_name}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate:.0e}"
        f"{'_ds' + str(args.downsample) if args.downsample else ''}"
    )

    wandb.init(
    project=project_name,
    name=run_name,
    config={
            "dataset_path": args.dataset_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "downsample": args.downsample,
            "embedding": "hex",
            "backbone": "bert-base-uncased",
            "trace_net": "fc",
            "fusion": "concat"
        }
    )

    # Load the dataset
    (X_profiling, Y_profiling, plaintexts_profiling), (X_attack, Y_attack, plaintexts_attack), profiling_mean, profiling_std = load_ascad_data(args.dataset_path, downsample=args.downsample)
    # Save the profiling mean and std
    np.save('profiling_mean.npy', profiling_mean)
    np.save('profiling_std.npy', profiling_std)
    print("✅ Profiling mean and std saved for consistent evaluation.")

    trace_length = X_profiling.shape[1]

    # Create datasets and DataLoaders with our custom collate function
    train_dataset = ASCADDataset(X_profiling, Y_profiling, plaintexts_profiling, tokenizer)
    test_dataset = ASCADDataset(X_attack, Y_attack, plaintexts_attack, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)

    # Initialize model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SCA_Model(bert_model, trace_length).to(DEVICE)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            traces = batch['trace'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                traces = batch['trace'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(input_ids, attention_mask, traces)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%), Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

    # Save trained model
    model_save_path = f"bert_sca_{os.path.basename(args.dataset_path).split('.')[0]}_seed{args.seed}_e{args.epochs}_lr{args.learning_rate:.0e}_s{args.downsample if args.downsample else 'full'}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save config
    config_save_path = model_save_path.replace(".pth", "_config.yaml")
    config_dict = {
        "dataset_path": args.dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "downsample": args.downsample,
        "embedding": "hex",  # Change if you use different embedding
        "backbone": "bert-base-uncased",
        "trace_net": "fc",
        "fusion": "concat",
        "key_byte": 0,
        "num_traces": 10000,
        "random_seed": args.seed,  # Add seed to config
        "profiling_mean_path": 'profiling_mean.npy',
        "profiling_std_path": 'profiling_std.npy'
    }

    import yaml
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"✅ Config saved to {config_save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT SCA model on ASCAD dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ASCAD dataset (.h5 file)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--downsample", type=int, default=None, help="Optional downsampling size of the profiling set")
    parser.add_argument("--seed", type=int, default=222, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set the random seed
    set_seed(args.seed)
    
    train_model(args)