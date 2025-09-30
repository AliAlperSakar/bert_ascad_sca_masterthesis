# === all_in_one_experiment.py ===
import argparse
import yaml
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CONFIG ----------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------------- EMBEDDERS ----------------
class HexStringEmbedder:
    def __init__(self, backbone):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
    def encode(self, pts):
        strings = [' '.join(f'{b:02x}' for b in pt) for pt in pts]
        encodings = self.tokenizer(strings, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        return encodings['input_ids'], encodings['attention_mask']

class ByteListEmbedder:
    def __init__(self, backbone):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
    def encode(self, pts):
        strings = [' '.join(str(b) for b in pt) for pt in pts]
        encodings = self.tokenizer(strings, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        return encodings['input_ids'], encodings['attention_mask']

class AsciiStringEmbedder:
    def __init__(self, backbone):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
    def encode(self, pts):
        strings = [''.join(chr(b) for b in pt) for pt in pts]
        encodings = self.tokenizer(strings, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        return encodings['input_ids'], encodings['attention_mask']

# ---------------- DATASET ----------------
class ASCADDataset(Dataset):
    def __init__(self, traces, labels, plaintexts, embedder):
        self.traces = traces
        self.labels = labels
        self.plaintexts = plaintexts
        self.embedder = embedder
    def __len__(self): return len(self.traces)
    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        pt = self.plaintexts[idx]
        input_ids, attn = self.embedder.encode([pt])
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attn.squeeze(0),
            'trace': torch.tensor(trace, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'trace': torch.stack([b['trace'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch])
    }

# ---------------- MODEL ----------------
class BERTSCAModel(nn.Module):
    def __init__(self, backbone='bert-base-uncased', fusion='concat', trace_net='fc', trace_length=1400):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        if trace_net == 'fc':
            self.trace_proc = nn.Linear(trace_length, 768)
        elif trace_net == 'cnn1d':
            self.trace_proc = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(768))
        else:
            self.trace_proc = None
        in_dim = 768 * 2 if self.trace_proc else 768
        self.cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_dim, 256))
    def forward(self, ids, mask, trace):
        bert_out = self.bert(input_ids=ids, attention_mask=mask).pooler_output
        if self.trace_proc:
            x = self.trace_proc(trace.unsqueeze(1)).squeeze(1) if len(trace.shape) == 2 else self.trace_proc(trace)
            out = torch.cat((bert_out, x), dim=1)
        else:
            out = bert_out
        return self.cls(out)

# ---------------- TRAIN ----------------
def train_model(cfg):
    (Xp, Yp, Pp), (Xa, Ya, Pa, Ka) = load_data(cfg['dataset_path'], cfg['downsample'])
    embedders = {
        'hex': HexStringEmbedder,
        'byte': ByteListEmbedder,
        'ascii': AsciiStringEmbedder
    }
    embedder = embedders[cfg['embedding']](cfg['backbone'])
    train_ds = ASCADDataset(Xp, Yp, Pp, embedder)
    test_ds = ASCADDataset(Xa, Ya, Pa, embedder)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], collate_fn=collate_fn)
    model = BERTSCAModel(cfg['backbone'], cfg['fusion'], cfg['trace_net'], trace_length=Xp.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.CrossEntropyLoss()
    wandb.init(project="bert-sca-unified", config=cfg)

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            trace = batch['trace'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            opt.zero_grad()
            preds = model(ids, mask, trace)
            loss = loss_fn(preds, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += preds.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        acc = 100. * correct / total
        wandb.log({"epoch": epoch+1, "train_loss": total_loss / len(train_loader), "train_acc": acc})
    model_path = f"model_{cfg['embedding']}_{cfg['trace_net']}_{cfg['fusion']}.pth"
    torch.save(model.state_dict(), model_path)
    return model, embedder, (Xa, Pa, Ya)

# ---------------- EVAL ----------------
AES_Sbox = np.array([0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
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
                     0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16])

def evaluate(model, embedder, X, P, K, key_byte, save_prefix):
    model.eval()
    input_ids, mask = embedder.encode(P)
    with torch.no_grad():
        logits = model(input_ids.to(DEVICE), mask.to(DEVICE), torch.tensor(X, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    ranks = np.zeros((len(X), 2))
    acc = np.zeros(256)
    for i in range(len(X)):
        pt = P[i][key_byte]
        real_k = K[i][key_byte]
        for k in range(256): acc[k] += probs[i][AES_Sbox[pt ^ k]]
        guess = np.argsort(acc)[::-1]
        rank = np.where(guess == real_k)[0][0]
        ranks[i] = [i, min(rank, ranks[i-1][1] if i else 256)]
    save_csv_plot(ranks, save_prefix)
    return ranks[-1][1]

def save_csv_plot(ranks, prefix):
    with open(f"{prefix}.csv", 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(["trace_index", "rank"])
        writer.writerows(ranks)
    plt.plot(ranks[:,0], ranks[:,1]); plt.title("Key Rank"); plt.xlabel("Trace"); plt.ylabel("Rank")
    plt.grid(); plt.savefig(f"{prefix}.png"); plt.close()

# ---------------- LOAD DATA ----------------
def load_data(path, downsample):
    with h5py.File(path, 'r') as f:
        Xp = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        Yp = np.array(f['Profiling_traces/labels'])
        Pp = np.array(f['Profiling_traces/metadata'][:]['plaintext'])

        Xa = np.array(f['Attack_traces/traces'], dtype=np.float32)
        Ya = np.array(f['Attack_traces/labels'])
        Pa = np.array(f['Attack_traces/metadata'][:]['plaintext'])

        metadata = f['Attack_traces']['metadata'][:]
        Ka = np.array([row['key'] for row in metadata])  # ✅ FIXED

        mean, std = np.mean(Xp, axis=0, keepdims=True), np.std(Xp, axis=0, keepdims=True)
        Xp, Xa = (Xp - mean) / std, (Xa - mean) / std

        if downsample:
            idxs = np.random.choice(len(Xp), downsample, replace=False)
            Xp, Yp, Pp = Xp[idxs], Yp[idxs], Pp[idxs]

    return (Xp, Yp, Pp), (Xa, Ya, Pa, Ka)



# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, embedder, (Xa, Pa, Ka) = train_model(cfg)
    final_rank = evaluate(model, embedder, Xa[:cfg['num_traces']], Pa[:cfg['num_traces']], Ka[:cfg['num_traces']], cfg['key_byte'], cfg['save_plot'].replace(".png", ""))
    wandb.log({"final_rank": final_rank})
    wandb.finish()

if __name__ == '__main__':
    main()

# Now just run:
# python all_in_one_experiment.py --config configs/config.yaml