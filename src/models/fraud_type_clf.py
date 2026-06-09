"""
欺诈类型多分类(任务升级)
==========================
背景:is_fraud 二分类对各模型而言过于简单(F1≈1.000),区分度不足。
本脚本在【欺诈样本】上构建更具挑战性的【欺诈类型 7 分类】任务:
  客服诈骗 / 银行诈骗 / 投资诈骗 / 钓鱼诈骗 / 彩票诈骗 / 绑架诈骗 / 身份盗窃
类别高度不均衡(客服 2360 → 身份盗窃 122),需要模型真正理解语义而非靠表层关键词。

同时训练并对比:
  - 传统 ML:TF-IDF + 逻辑回归 / 线性 SVM(class_weight=balanced)
  - 预训练模型:中文 RoBERTa / MacBERT 微调

输出:
  - outputs/results/fraud_type_traditional.json
  - outputs/results/fraud_type_<tag>.json(BERT)
  - 保存 BERT 权重 outputs/models/ftype_<tag>/,标签映射 outputs/models/ftype_label_map.json
  - 测试集 macro/weighted P/R/F1、每类 P/R/F1、混淆矩阵

用法:
  python src/models/fraud_type_clf.py --mode traditional
  python src/models/fraud_type_clf.py --mode bert --model <本地路径> --tag macbert
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, f1_score)

PROC = "data/processed"
RES = "outputs/results"
MODELS = "outputs/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# 固定标签顺序(按训练集频次降序,便于阅读)
LABELS = ["客服诈骗", "银行诈骗", "投资诈骗", "钓鱼诈骗", "彩票诈骗", "绑架诈骗", "身份盗窃"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_fraud_split(name):
    """仅取欺诈样本(label==1 且 fraud_type 有效),返回 text 与 fraud_type 整数标签。"""
    d = pd.read_csv(os.path.join(PROC, f"{name}.csv"))
    d = d[(d["label"] == 1) & (d["fraud_type"].isin(LABELS))].copy()
    d["y"] = d["fraud_type"].map(LABEL2ID).astype(int)
    return d.reset_index(drop=True)


def full_metrics(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f_macro,
        "weighted_precision": p_w, "weighted_recall": r_w, "weighted_f1": f_w,
        "per_class": classification_report(
            y_true, y_pred, labels=list(range(len(LABELS))),
            target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


# ===================== 传统 ML =====================
def run_traditional():
    import jieba
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    jieba.setLogLevel(20)

    train = load_fraud_split("train")
    val = load_fraud_split("val")
    test = load_fraud_split("test")
    print(f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)} (欺诈样本)")
    print("类别分布(train):", train["fraud_type"].value_counts().to_dict())

    tok = lambda t: " ".join(jieba.cut(str(t)))
    Xtr_txt = train["text"].apply(tok)
    Xval_txt = val["text"].apply(tok)
    Xte_txt = test["text"].apply(tok)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_txt)
    Xval = vec.transform(Xval_txt)
    Xte = vec.transform(Xte_txt)
    ytr, yval, yte = train["y"].values, val["y"].values, test["y"].values

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=8.0, class_weight="balanced"),
        "LinearSVM": LinearSVC(C=1.0, class_weight="balanced"),
    }
    results = {}
    # 模型选择在【验证集】上进行,test 仅用于最终报告(避免 test-set selection)
    best_val_f1, best_name, best_clf = -1, None, None
    for name, clf in models.items():
        print(f"\n训练 {name} ...")
        clf.fit(Xtr, ytr)
        val_f1 = f1_score(yval, clf.predict(Xval), average="macro", zero_division=0)
        m = full_metrics(yte, clf.predict(Xte))
        m["val_macro_f1"] = val_f1
        results[name] = m
        print(f"  [val] macro-F1={val_f1:.4f}   [test] acc={m['accuracy']:.4f}  "
              f"macro-F1={m['macro_f1']:.4f}  weighted-F1={m['weighted_f1']:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_name, best_clf = val_f1, name, clf

    results["_meta"] = {"labels": LABELS, "n_train": len(train), "n_val": len(val),
                        "n_test": len(test), "best": best_name,
                        "selected_by": "val_macro_f1"}
    with open(os.path.join(RES, "fraud_type_traditional.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    joblib.dump(vec, os.path.join(MODELS, "ftype_tfidf_vectorizer.joblib"))
    joblib.dump(best_clf, os.path.join(MODELS, f"ftype_traditional_{best_name}.joblib"))
    with open(os.path.join(MODELS, "ftype_label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID}, f, ensure_ascii=False, indent=2)
    print(f"\n最优传统模型(按验证集选): {best_name} "
          f"(val macro-F1={best_val_f1:.4f}, test macro-F1={results[best_name]['macro_f1']:.4f}),已保存。")


# ===================== BERT 微调 =====================
def run_bert(model_path, tag, epochs, batch_size, lr, max_len):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42); np.random.seed(42)
    import random; random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train = load_fraud_split("train")
    val = load_fraud_split("val")
    test = load_fraud_split("test")
    print(f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")
    print("类别分布(train):", train["fraud_type"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts); self.labels = list(labels)
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(self.labels[i], dtype=torch.long)}

    g = torch.Generator(); g.manual_seed(42)
    tl = DataLoader(DS(train["text"], train["y"]), batch_size=batch_size,
                    shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(DS(val["text"], val["y"]), batch_size=64, num_workers=0)
    el = DataLoader(DS(test["text"], test["y"]), batch_size=64, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(LABELS)).to(DEVICE)

    # 类别加权交叉熵(缓解不均衡)
    counts = train["y"].value_counts().reindex(range(len(LABELS))).fillna(0).values
    weights = torch.tensor((counts.sum() / (len(LABELS) * np.clip(counts, 1, None))),
                           dtype=torch.float).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(tl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    def evaluate(loader):
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for b in loader:
                logits = model(input_ids=b["input_ids"].to(DEVICE),
                               attention_mask=b["attention_mask"].to(DEVICE)).logits
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                golds.extend(b["labels"].numpy().tolist())
        return preds, golds

    save_dir = os.path.join(MODELS, f"ftype_{tag}")
    best_f1 = -1
    history = []
    for epoch in range(epochs):
        model.train(); total = 0
        for step, b in enumerate(tl):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(input_ids=b["input_ids"].to(DEVICE),
                               attention_mask=b["attention_mask"].to(DEVICE)).logits
                loss = criterion(logits, b["labels"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(tl)} loss {loss.item():.4f}")
        vp, vg = evaluate(vl)
        vmf1 = f1_score(vg, vp, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": total / len(tl), "val_macro_f1": vmf1})
        print(f"[epoch {epoch+1}] loss={total/len(tl):.4f} val_macro_f1={vmf1:.4f}")
        if vmf1 > best_f1:
            best_f1 = vmf1
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)

    print("加载最优权重在测试集评估...")
    best = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)
    model = best
    tp, tg = evaluate(el)
    m = full_metrics(tg, tp)
    print(f"[TEST] acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  "
          f"weighted-F1={m['weighted_f1']:.4f}")

    out = {"model": model_path, "tag": tag, "labels": LABELS,
           "args": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len},
           "history": history, "test": m}
    with open(os.path.join(RES, f"fraud_type_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("结果已保存。")


# ===================== 深度学习(TextCNN / BiLSTM, 字符级) =====================
def run_deep(arch, tag, epochs, batch_size, lr, max_len=400):
    import sys as _sys
    import torch
    from torch.utils.data import DataLoader
    _sys.path.insert(0, "src/models")
    from deep_models import TextCNN, BiLSTMAttn, CharDataset, build_vocab
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42); np.random.seed(42)
    import random; random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train = load_fraud_split("train")
    val = load_fraud_split("val")
    test = load_fraud_split("test")
    print(f"训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")

    vocab = build_vocab(train["text"])
    print("词表大小:", len(vocab))
    g = torch.Generator(); g.manual_seed(42)
    tl = DataLoader(CharDataset(train["text"], train["y"], vocab, max_len),
                    batch_size=batch_size, shuffle=True, generator=g)
    vl = DataLoader(CharDataset(val["text"], val["y"], vocab, max_len), batch_size=128)
    el = DataLoader(CharDataset(test["text"], test["y"], vocab, max_len), batch_size=128)

    if arch == "textcnn":
        model = TextCNN(len(vocab), num_classes=len(LABELS)).to(DEVICE)
    else:
        model = BiLSTMAttn(len(vocab), num_classes=len(LABELS)).to(DEVICE)

    counts = train["y"].value_counts().reindex(range(len(LABELS))).fillna(0).values
    weights = torch.tensor((counts.sum() / (len(LABELS) * np.clip(counts, 1, None))),
                           dtype=torch.float).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate(loader):
        model.eval(); preds, golds = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(DEVICE))
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                golds.extend(y.numpy().tolist())
        return preds, golds

    save_path = os.path.join(MODELS, f"ftype_deep_{tag}.pt")
    best_f1 = -1; history = []
    for epoch in range(epochs):
        model.train(); total = 0
        for x, y in tl:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward(); optimizer.step(); total += loss.item()
        vp, vg = evaluate(vl)
        vmf1 = f1_score(vg, vp, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": total / len(tl), "val_macro_f1": vmf1})
        print(f"[epoch {epoch+1}] loss={total/len(tl):.4f} val_macro_f1={vmf1:.4f}")
        if vmf1 > best_f1:
            best_f1 = vmf1
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    tp, tg = evaluate(el)
    m = full_metrics(tg, tp)
    print(f"[TEST] acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  "
          f"weighted-F1={m['weighted_f1']:.4f}")
    out = {"arch": arch, "tag": tag, "labels": LABELS,
           "args": {"epochs": epochs, "batch_size": batch_size, "lr": lr},
           "history": history, "test": m}
    with open(os.path.join(RES, f"fraud_type_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("结果已保存。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["traditional", "bert", "deep"], required=True)
    ap.add_argument("--model", default=None, help="BERT 本地路径或 HF 名")
    ap.add_argument("--arch", choices=["textcnn", "bilstm"], default="textcnn")
    ap.add_argument("--tag", default="bert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    if args.mode == "traditional":
        run_traditional()
    elif args.mode == "deep":
        run_deep(args.arch, args.tag, args.epochs, args.batch_size, args.lr)
    else:
        assert args.model, "--mode bert 需指定 --model"
        run_bert(args.model, args.tag, args.epochs, args.batch_size, args.lr, args.max_len)


if __name__ == "__main__":
    main()
