"""
深度学习基线:TextCNN / BiLSTM(字符级,随机初始化嵌入)
- 字符级词表(中文按字切分,简单稳健)
- TextCNN(多尺度卷积) 与 BiLSTM(+注意力) 两种结构
- 输出测试集 Accuracy/Precision/Recall/F1,保存模型与词表
用法: python src/models/deep_models.py --arch textcnn --tag textcnn
"""
import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PROC = "data/processed"
RES = "outputs/results"
MODELS = "outputs/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, UNK = 0, 1


def build_vocab(texts, min_freq=2, max_size=30000):
    counter = Counter()
    for t in texts:
        counter.update(list(str(t)))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for ch, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, max_len=400):
    ids = [vocab.get(ch, UNK) for ch in str(text)[:max_len]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids


class CharDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=400):
        self.x = [encode(t, vocab, max_len) for t in texts]
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.x[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)  # B,E,L
        feats = [torch.relu(conv(e)).max(dim=2)[0] for conv in self.convs]
        out = self.dropout(torch.cat(feats, dim=1))
        return self.fc(out)


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=dropout)
        self.attn = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        e = self.embed(x)
        h, _ = self.lstm(e)  # B,L,2H
        scores = self.attn(h).masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1)
        ctx = (h * w).sum(dim=1)
        return self.fc(self.dropout(ctx))


def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            golds.extend(y.numpy().tolist())
    return {
        "accuracy": accuracy_score(golds, preds),
        "precision": precision_score(golds, preds, zero_division=0),
        "recall": recall_score(golds, preds, zero_division=0),
        "f1": f1_score(golds, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(golds, preds).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["textcnn", "bilstm"], default="textcnn")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=400)
    args = ap.parse_args()
    tag = args.tag or args.arch

    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train = pd.read_csv(os.path.join(PROC, "train.csv"))
    val = pd.read_csv(os.path.join(PROC, "val.csv"))
    test = pd.read_csv(os.path.join(PROC, "test.csv"))

    vocab = build_vocab(train["text"])
    print("词表大小:", len(vocab))

    train_ds = CharDataset(train["text"], train["label"], vocab, args.max_len)
    val_ds = CharDataset(val["text"], val["label"], vocab, args.max_len)
    test_ds = CharDataset(test["text"], test["label"], vocab, args.max_len)
    g = torch.Generator(); g.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    if args.arch == "textcnn":
        model = TextCNN(len(vocab)).to(DEVICE)
    else:
        model = BiLSTMAttn(len(vocab)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1
    save_path = os.path.join(MODELS, f"deep_{tag}.pt")
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_m = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_loader), "val": val_m})
        print(f"[epoch {epoch+1}] loss={total/len(train_loader):.4f} "
              f"val_f1={val_m['f1']:.4f} val_acc={val_m['accuracy']:.4f}")
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    test_m = evaluate(model, test_loader)
    print("[TEST]", "  ".join(f"{k}={v:.4f}" for k, v in test_m.items() if k != "confusion_matrix"))

    with open(os.path.join(MODELS, f"deep_{tag}_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(RES, f"deep_{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": args.arch, "tag": tag, "args": vars(args),
                   "history": history, "test": test_m}, f, ensure_ascii=False, indent=2)
    print("结果已保存。")


if __name__ == "__main__":
    main()
