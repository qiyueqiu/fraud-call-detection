#!/usr/bin/env bash
# ============================================================
# 欺诈通话检测 + Fraud-R1 鲁棒性分析 —— 一键复现脚本
# ============================================================
set -e
cd "$(dirname "$0")"

echo "[1/6] 数据预处理..."
python src/data/preprocess.py

echo "[2/6] 探索性数据分析(EDA 图)..."
python src/data/eda.py

echo "[3/6] 训练传统机器学习模型(TF-IDF + LR/SVM/RF/NB)..."
python src/models/traditional_ml.py

echo "[4/6] 训练深度模型与预训练模型..."
python src/models/deep_models.py --arch textcnn --tag textcnn
python src/models/deep_models.py --arch bilstm  --tag bilstm
python src/models/bert_finetune.py --model hfl/chinese-roberta-wwm-ext --tag roberta --epochs 3
python src/models/bert_finetune.py --model hfl/chinese-macbert-base     --tag macbert --epochs 3

echo "[5/6] Fraud-R1 风格语义改写(需设置 DEEPSEEK_API_KEY)..."
python src/rewrite/fraud_r1_rewrite.py --strategy all --sample 400

echo "[6/6] 鲁棒性评估 + 结果可视化..."
python src/rewrite/evaluate_robustness.py
python src/utils/plot_results.py
python src/utils/summary_table.py

echo "全部完成。结果见 outputs/ ,论文见 paper/main.pdf"
