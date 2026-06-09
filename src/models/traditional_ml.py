"""
传统机器学习基线:TF-IDF + 多种分类器
- 中文分词(jieba) + TF-IDF 特征
- 逻辑回归 / 线性SVM / 随机森林 / 多项式朴素贝叶斯
- 输出各模型在测试集上的 Accuracy/Precision/Recall/F1
- 保存最优传统模型与向量器,供鲁棒性测试复用
"""
import os
import json
import joblib
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

jieba.setLogLevel(20)
PROC = "data/processed"
RES = "outputs/results"
MODELS = "outputs/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def tokenize(text: str) -> str:
    return " ".join(jieba.cut(str(text)))


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    train = pd.read_csv(os.path.join(PROC, "train.csv"))
    val = pd.read_csv(os.path.join(PROC, "val.csv"))
    test = pd.read_csv(os.path.join(PROC, "test.csv"))
    # 合并 train+val 用于最终训练(验证集用于超参可在此略)
    train_all = pd.concat([train, val], ignore_index=True)

    print("中文分词中...")
    X_train_txt = train_all["text"].apply(tokenize)
    X_test_txt = test["text"].apply(tokenize)
    y_train = train_all["label"].values
    y_test = test["label"].values

    print("构建 TF-IDF 特征(1-2 gram)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    print("特征维度:", X_train.shape)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=4.0, n_jobs=-1),
        "LinearSVM": LinearSVC(C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=0.3),
    }

    results = {}
    best_f1, best_name, best_model = -1, None, None
    for name, clf in models.items():
        print(f"\n训练 {name} ...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        m = evaluate(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results[name] = {**m, "confusion_matrix": cm}
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        if m["f1"] > best_f1:
            best_f1, best_name, best_model = m["f1"], name, clf

    # 保存结果与最优模型
    with open(os.path.join(RES, "traditional_ml_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    joblib.dump(vectorizer, os.path.join(MODELS, "tfidf_vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODELS, f"best_traditional_{best_name}.joblib"))
    print(f"\n最优传统模型: {best_name} (F1={best_f1:.4f}),已保存。")


if __name__ == "__main__":
    main()
