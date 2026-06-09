"""
欺诈类型多分类结果可视化
==========================
基于 outputs/results/fraud_type_*.json 生成:
  1. 三模型 macro/weighted-F1 与准确率对比(说明任务难度,对照二分类的近完美)
  2. 各模型逐类 F1 热力图(揭示哪些欺诈类型易混淆)
  3. 最优模型(按 macro-F1)的 7×7 混淆矩阵
输出至 outputs/figures/{fraud_type_overview,fraud_type_perclass,fraud_type_confusion}.{pdf,png}
"""
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'utils'))
from figio import savefig_dual
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
matplotlib.use("Agg")


def setup_font():
    p = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
        plt.rcParams["axes.unicode_minus"] = False
setup_font()

RES = "outputs/results"
FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)
LABELS = ["客服诈骗", "银行诈骗", "投资诈骗", "钓鱼诈骗", "彩票诈骗", "绑架诈骗", "身份盗窃"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_all():
    """返回 {模型名: metrics dict(含 per_class, confusion_matrix, macro_f1...)}"""
    out = {}
    # 传统(取最优 LogisticRegression)
    tp = os.path.join(RES, "fraud_type_traditional.json")
    if os.path.exists(tp):
        d = json.load(open(tp, encoding="utf-8"))
        out["TF-IDF+LR"] = d["LogisticRegression"]
    for tag, name in [("textcnn", "TextCNN"), ("roberta", "RoBERTa"), ("macbert", "MacBERT")]:
        p = os.path.join(RES, f"fraud_type_{tag}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))["test"]
    return out


def plot_overview(data):
    models = list(data.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    metric_cn = ["准确率", "Macro-F1", "Weighted-F1"]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(models):
        vals = [data[m][k] for k in metrics]
        bars = ax.bar(x + i * width, vals, width, label=m, color=PALETTE[i % len(PALETTE)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_cn)
    ax.set_ylabel("分数")
    ax.set_ylim(0.6, 0.85)
    ax.set_title("欺诈类型 7 分类性能对比(对照二分类 F1≈1.000,本任务显著更难)")
    ax.legend(ncol=len(models), fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "fraud_type_overview.png"))
    plt.close()
    print("saved fraud_type_overview.png")


def plot_perclass(data):
    models = list(data.keys())
    mat = np.array([[data[m]["per_class"][lab]["f1-score"] for lab in LABELS]
                    for m in models])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color="black")
    ax.set_title("各模型在每类欺诈类型上的 F1(绿=易区分,红=易混淆)")
    fig.colorbar(im, ax=ax, label="F1", fraction=0.025)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "fraud_type_perclass.png"))
    plt.close()
    print("saved fraud_type_perclass.png")


def plot_confusion(data):
    # 选 macro-F1 最高的模型
    best = max(data.keys(), key=lambda m: data[m]["macro_f1"])
    cm = np.array(data[best]["confusion_matrix"], dtype=float)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = cmn[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("预测类型"); ax.set_ylabel("真实类型")
    ax.set_title(f"{best} 欺诈类型混淆矩阵(行归一化)\nmacro-F1={data[best]['macro_f1']:.3f}")
    fig.colorbar(im, ax=ax, label="比例", fraction=0.046)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "fraud_type_confusion.png"))
    plt.close()
    print(f"saved fraud_type_confusion.png (best={best})")


if __name__ == "__main__":
    data = load_all()
    print("载入模型:", list(data.keys()))
    plot_overview(data)
    plot_perclass(data)
    plot_confusion(data)
    print("\n多分类图表已生成至", FIG)
