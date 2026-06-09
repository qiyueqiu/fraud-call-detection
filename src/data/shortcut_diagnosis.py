"""
数据集捷径(shortcut)与泄漏诊断
================================
解释"为何各模型在原始测试集上接近完美"这一现象的根因。
分析:
  1. train/test 完全重复对话(数据泄漏)
  2. 训练集内部重复度
  3. 单一关键词规则的判别力(仅靠"是否含若干欺诈关键词"能否近乎完美分类)
  4. 正常/欺诈通话的关键词覆盖差异
  5. 最具判别力的词(逻辑回归权重 Top-N)
输出: outputs/results/shortcut_diagnosis.json
"""
import os
import json
import joblib
import jieba
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

jieba.setLogLevel(20)
PROC = "data/processed"
RES = "outputs/results"
MODELS = "outputs/models"
os.makedirs(RES, exist_ok=True)

KEYWORDS = ["点击", "链接", "验证码", "转账", "二维码", "银行卡", "安全账户",
            "冻结", "汇款", "退款", "密码", "输入"]

train = pd.read_csv(os.path.join(PROC, "train.csv"))
val = pd.read_csv(os.path.join(PROC, "val.csv"))
test = pd.read_csv(os.path.join(PROC, "test.csv"))
train_all = pd.concat([train, val], ignore_index=True)

out = {}

# 1. train/test 泄漏
inter = set(train_all["text"]) & set(test["text"])
out["train_test_overlap"] = {
    "n_overlap": len(inter),
    "test_size": len(test),
    "overlap_ratio": round(len(inter) / len(test), 4),
}

# 2. 训练集内部重复
out["train_dedup"] = {
    "unique": int(train_all["text"].nunique()),
    "total": len(train_all),
    "dup_ratio": round(1 - train_all["text"].nunique() / len(train_all), 4),
}

# 3. 单一关键词规则判别力
def has_kw(s):
    return int(any(k in str(s) for k in KEYWORDS))
test_kw = test["text"].apply(has_kw)
out["keyword_rule"] = {
    "keywords": KEYWORDS,
    "accuracy": round(accuracy_score(test["label"], test_kw), 4),
    "f1": round(f1_score(test["label"], test_kw), 4),
    "fraud_has_kw_ratio": round(test[test.label == 1]["text"].apply(has_kw).mean(), 4),
    "normal_has_kw_ratio": round(test[test.label == 0]["text"].apply(has_kw).mean(), 4),
}

# 4. 各单关键词在两类中的覆盖
per_kw = {}
for k in KEYWORDS:
    f_rate = test[test.label == 1]["text"].str.contains(k, regex=False).mean()
    n_rate = test[test.label == 0]["text"].str.contains(k, regex=False).mean()
    per_kw[k] = {"fraud": round(f_rate, 4), "normal": round(n_rate, 4),
                 "gap": round(f_rate - n_rate, 4)}
out["per_keyword_coverage"] = per_kw

# 5. 逻辑回归 Top 判别词
try:
    vec = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.joblib"))
    clf = joblib.load([os.path.join(MODELS, f) for f in os.listdir(MODELS)
                       if f.startswith("best_traditional_")][0])
    if hasattr(clf, "coef_"):
        feats = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]
        top_fraud = feats[np.argsort(coef)[-15:][::-1]].tolist()
        top_normal = feats[np.argsort(coef)[:15]].tolist()
        out["top_discriminative_words"] = {
            "toward_fraud": top_fraud, "toward_normal": top_normal}
except Exception as e:
    out["top_discriminative_words"] = {"error": str(e)[:80]}

with open(os.path.join(RES, "shortcut_diagnosis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(json.dumps(out, ensure_ascii=False, indent=2))
