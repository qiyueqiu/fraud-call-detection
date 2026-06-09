"""
BERT 系预训练模型微调:中文欺诈通话二分类
- 支持 chinese-roberta-wwm-ext / chinese-bert-wwm-ext / chinese-macbert-base 等
- torch 原生训练循环(兼容 transformers 5.x)
- 输出测试集 Accuracy/Precision/Recall/F1,保存最优模型权重
用法: python src/models/bert_finetune.py --model hfl/chinese-roberta-wwm-ext --tag roberta
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PROC = "data/processed"
RES = "outputs/results"
MODELS = "outputs/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
            golds.extend(batch["labels"].numpy().tolist())
    return {
        "accuracy": accuracy_score(golds, preds),
        "precision": precision_score(golds, preds, zero_division=0),
        "recall": recall_score(golds, preds, zero_division=0),
        "f1": f1_score(golds, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(golds, preds).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--tag", default="roberta")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

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

    print(f"加载模型 {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).to(DEVICE)

    train_ds = DialogueDataset(train["text"], train["label"], tokenizer, args.max_len)
    val_ds = DialogueDataset(val["text"], val["label"], tokenizer, args.max_len)
    test_ds = DialogueDataset(test["text"], test["label"], tokenizer, args.max_len)

    g = torch.Generator(); g.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_f1 = -1
    save_dir = os.path.join(MODELS, f"bert_{args.tag}")
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = out.loss
            assert loss is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        val_m = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader), "val": val_m})
        print(f"[epoch {epoch+1}] train_loss={total_loss/len(train_loader):.4f} "
              f"val_f1={val_m['f1']:.4f} val_acc={val_m['accuracy']:.4f}")
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # 用最优权重在测试集评估
    print("加载最优权重并在测试集评估...")
    best_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)
    test_m = evaluate(best_model, test_loader)
    print("[TEST]", "  ".join(f"{k}={v:.4f}" for k, v in test_m.items() if k != "confusion_matrix"))

    out = {"model": args.model, "tag": args.tag, "args": vars(args),
           "history": history, "test": test_m}
    with open(os.path.join(RES, f"bert_{args.tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("结果已保存。")


if __name__ == "__main__":
    main()
