"""
错误案例定性分析
==================
聚焦"对抗规避攻击成功骗过检测器"的样本(false negative):
即原始能被正确识别为欺诈、改写后被误判为正常的样本。
分析其改写前后的关键词变化、文本特征,揭示模型脆弱的具体机制。
输出: outputs/results/error_cases.json + 控制台可读案例
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

jieba.setLogLevel(20)
PROC = "data/processed"
REWRITTEN = "data/rewritten"
MODELS = "outputs/models"
RES = "outputs/results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEYWORDS = ["点击", "链接", "验证码", "转账", "二维码", "银行卡", "冻结",
            "汇款", "退款", "密码", "输入", "账户"]


def kw_count(s):
    return sum(str(s).count(k) for k in KEYWORDS)


# --- BiLSTM 预测(最脆弱模型)---
import sys
sys.path.insert(0, "src/models")
from deep_models import BiLSTMAttn, encode


def load_bilstm():
    with open(os.path.join(MODELS, "deep_bilstm_vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    model = BiLSTMAttn(len(vocab))
    model.load_state_dict(torch.load(os.path.join(MODELS, "deep_bilstm.pt")))
    model.to(DEVICE).eval()
    return model, vocab


def predict_bilstm(texts, model, vocab):
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            x = torch.tensor([encode(t, vocab, 400) for t in texts[i:i+128]]).to(DEVICE)
            p = F.softmax(model(x), dim=-1)
            preds.extend(p.argmax(-1).cpu().numpy().tolist())
            probs.extend(p[:, 1].cpu().numpy().tolist())
    return np.array(preds), np.array(probs)


def main():
    model, vocab = load_bilstm()
    strat = "evasion"  # 最有效的规避策略
    df = pd.read_csv(os.path.join(REWRITTEN, f"test_{strat}.csv"))

    pred_o, prob_o = predict_bilstm(df["original_text"].tolist(), model, vocab)
    pred_r, prob_r = predict_bilstm(df["text"].tolist(), model, vocab)

    # 攻击成功:原始判对(欺诈)、改写后判错(正常)
    flipped = (pred_o == 1) & (pred_r == 0)
    n_flip = int(flipped.sum())
    print(f"策略 {strat}: 攻击成功翻转 {n_flip} / {len(df)} 条")

    cases = []
    for i in np.where(flipped)[0]:
        cases.append({
            "fraud_type": df.iloc[i].get("fraud_type", ""),
            "kw_before": kw_count(df.iloc[i]["original_text"]),
            "kw_after": kw_count(df.iloc[i]["text"]),
            "prob_before": round(float(prob_o[i]), 4),
            "prob_after": round(float(prob_r[i]), 4),
            "original_excerpt": df.iloc[i]["original_text"][:200],
            "rewritten_excerpt": df.iloc[i]["text"][:200],
        })

    summary = {
        "strategy": strat,
        "n_flipped": n_flip,
        "total": len(df),
        "avg_kw_before": round(np.mean([c["kw_before"] for c in cases]), 2) if cases else 0,
        "avg_kw_after": round(np.mean([c["kw_after"] for c in cases]), 2) if cases else 0,
        "avg_prob_before": round(np.mean([c["prob_before"] for c in cases]), 4) if cases else 0,
        "avg_prob_after": round(np.mean([c["prob_after"] for c in cases]), 4) if cases else 0,
        "cases": cases,
    }
    with open(os.path.join(RES, "error_cases.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"翻转样本平均关键词: {summary['avg_kw_before']} -> {summary['avg_kw_after']}")
    print(f"翻转样本平均欺诈置信度: {summary['avg_prob_before']} -> {summary['avg_prob_after']}")
    print("\n=== 典型案例 ===")
    for c in cases[:3]:
        print(f"\n[{c['fraud_type']}] 关键词 {c['kw_before']}->{c['kw_after']}, "
              f"置信度 {c['prob_before']}->{c['prob_after']}")
        print("原始:", c["original_excerpt"][:120])
        print("规避:", c["rewritten_excerpt"][:120])


if __name__ == "__main__":
    main()
