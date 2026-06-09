"""
鲁棒性评估 v2:在 Fraud-R1 改写测试集上评测已训练分类器(不重新训练)
=====================================================================
评估两类指标:
  1. 欺诈检出率(Detection Rate)= 对欺诈样本预测为"欺诈"的比例(= Recall)
     检出率下降 = 攻击成功率(ASR);仍正确识别的比例 = 防御成功率(DSR, 对照 Fraud-R1)
  2. 平均欺诈置信度(Mean Fraud Probability)= 模型对"欺诈"类的平均预测概率
     即使标签未翻转,置信度下降也揭示鲁棒性裂缝

策略分两组:
  - 诱导增强组(inducement): trust/urgency/emotion/authority/paraphrase —— 增强对"人"的说服力
  - 对抗规避组(evasion):     evasion/benign_disguise/code_switch  —— 主动隐藏特征以绕过"检测器"

输出: outputs/results/robustness_results.json
"""
import os
import json
import pickle
import joblib
import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

jieba.setLogLevel(20)
PROC = "data/processed"
REWRITTEN = "data/rewritten"
MODELS = "outputs/models"
RES = "outputs/results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INDUCEMENT = ["trust", "urgency", "emotion", "authority", "paraphrase"]
EVASION = ["evasion", "benign_disguise", "code_switch"]
ALL_STRATS = INDUCEMENT + EVASION


# ---------- 预测(返回 预测标签 与 欺诈概率)----------
_cache = {}


def get_traditional():
    if "trad" not in _cache:
        vec = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.joblib"))
        cand = [f for f in os.listdir(MODELS) if f.startswith("best_traditional_")][0]
        clf = joblib.load(os.path.join(MODELS, cand))
        name = cand.replace("best_traditional_", "").replace(".joblib", "")
        _cache["trad"] = (vec, clf, name)
    return _cache["trad"]


def predict_traditional(texts):
    vec, clf, _ = get_traditional()
    X = vec.transform([" ".join(jieba.cut(str(t))) for t in texts])
    pred = clf.predict(X)
    # 概率:LogReg 有 predict_proba;SVM 用 decision_function 经 sigmoid 近似
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
    else:
        d = clf.decision_function(X)
        prob = 1 / (1 + np.exp(-d))
    return np.array(pred), np.array(prob)


def predict_deep(texts, tag, max_len=400):
    import sys
    sys.path.insert(0, "src/models")
    from deep_models import TextCNN, BiLSTMAttn, encode
    key = f"deep_{tag}"
    if key not in _cache:
        with open(os.path.join(MODELS, f"deep_{tag}_vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        model = TextCNN(len(vocab)) if tag == "textcnn" else BiLSTMAttn(len(vocab))
        model.load_state_dict(torch.load(os.path.join(MODELS, f"deep_{tag}.pt")))
        model.to(DEVICE).eval()
        _cache[key] = (model, vocab)
    model, vocab = _cache[key]
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            batch = texts[i:i + 128]
            x = torch.tensor([encode(t, vocab, max_len) for t in batch]).to(DEVICE)
            p = F.softmax(model(x), dim=-1)
            preds.extend(p.argmax(-1).cpu().numpy().tolist())
            probs.extend(p[:, 1].cpu().numpy().tolist())
    return np.array(preds), np.array(probs)


def predict_bert(texts, tag, max_len=256):
    key = f"bert_{tag}"
    if key not in _cache:
        path = os.path.join(MODELS, f"bert_{tag}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        _cache[key] = (model, tok)
    model, tok = _cache[key]
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = list(texts[i:i + 64])
            enc = tok(batch, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEVICE)
            p = F.softmax(model(**enc).logits, dim=-1)
            preds.extend(p.argmax(-1).cpu().numpy().tolist())
            probs.extend(p[:, 1].cpu().numpy().tolist())
    return np.array(preds), np.array(probs)


def run(texts, kind, tag):
    if kind == "traditional":
        return predict_traditional(texts)
    if kind == "deep":
        return predict_deep(texts, tag)
    return predict_bert(texts, tag)


def main():
    sample_df = pd.read_csv(os.path.join(REWRITTEN, "test_trust.csv"))
    orig_texts = sample_df["original_text"].tolist()
    n = len(orig_texts)
    print(f"评估样本数(欺诈): {n}")

    bert_tags = [t.replace("bert_", "") for t in os.listdir(MODELS)
                 if t.startswith("bert_") and os.path.isdir(os.path.join(MODELS, t))]
    deep_tags = [f.replace("deep_", "").replace("_vocab.pkl", "")
                 for f in os.listdir(MODELS) if f.endswith("_vocab.pkl")]

    _, _, trad_name = get_traditional()
    models = {f"TF-IDF+{trad_name}": ("traditional", trad_name)}
    name_map = {"textcnn": "TextCNN", "bilstm": "BiLSTM",
                "roberta": "RoBERTa", "macbert": "MacBERT"}
    for tag in deep_tags:
        models[name_map.get(tag, tag)] = ("deep", tag)
    for tag in bert_tags:
        models[name_map.get(tag, tag)] = ("bert", tag)
    print("参评模型:", list(models.keys()))

    avail = [s for s in ALL_STRATS
             if os.path.exists(os.path.join(REWRITTEN, f"test_{s}.csv"))]
    print("可用策略:", avail)

    results = {}
    for mname, (kind, tag) in models.items():
        print(f"\n--- 模型 {mname} ---")
        row = {}
        pred, prob = run(orig_texts, kind, tag)
        row["original"] = {"detection_rate": float(np.mean(pred == 1)),
                           "mean_fraud_prob": float(np.mean(prob))}
        print(f"  original   : 检出率 {row['original']['detection_rate']:.4f}  "
              f"置信度 {row['original']['mean_fraud_prob']:.4f}")
        for strat in avail:
            df = pd.read_csv(os.path.join(REWRITTEN, f"test_{strat}.csv"))
            pred, prob = run(df["text"].tolist(), kind, tag)
            dr = float(np.mean(pred == 1))
            mp = float(np.mean(prob))
            row[strat] = {"detection_rate": dr, "mean_fraud_prob": mp}
            d_dr = row["original"]["detection_rate"] - dr
            d_mp = row["original"]["mean_fraud_prob"] - mp
            print(f"  {strat:13s}: 检出率 {dr:.4f} (ΔDR {d_dr:+.4f})  "
                  f"置信度 {mp:.4f} (Δprob {d_mp:+.4f})")
        results[mname] = row

    out = {
        "n_samples": n,
        "inducement_strategies": [s for s in INDUCEMENT if s in avail],
        "evasion_strategies": [s for s in EVASION if s in avail],
        "metrics": ["detection_rate (=recall on fraud)", "mean_fraud_prob"],
        "results": results,
    }
    with open(os.path.join(RES, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n鲁棒性结果已保存。")


if __name__ == "__main__":
    main()
