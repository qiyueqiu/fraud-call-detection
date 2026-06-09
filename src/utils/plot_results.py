"""
实验结果可视化
================
基于 outputs/results/ 下的 JSON 结果生成论文配图:
  1. 各模型原始测试集性能对比柱状图
  2. 鲁棒性热力图(模型 × 策略 的检出率)
  3. 检出率下降(ASR)分组柱状图:诱导增强组 vs 对抗规避组
  4. 置信度变化对比图
  5. 各模型混淆矩阵(原始测试集)
"""
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, _os.path.dirname(__file__))
from figio import savefig_dual
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")


def setup_font():
    for p in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=p).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return
setup_font()

RES = "outputs/results"
FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

STRAT_CN = {
    "original": "原始", "trust": "信任建立", "urgency": "紧迫感",
    "emotion": "情感操纵", "authority": "权威伪装", "paraphrase": "语义改写",
    "evasion": "关键词规避", "benign_disguise": "良性伪装", "code_switch": "谐音隐语",
}
MODEL_ORDER = ["TF-IDF+LogisticRegression", "TextCNN", "BiLSTM", "RoBERTa", "MacBERT"]
MODEL_CN = {"TF-IDF+LogisticRegression": "TF-IDF+LR"}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ========== 1. 原始测试集性能对比 ==========
def plot_baseline_performance():
    files = {
        "TF-IDF+LogisticRegression": ("traditional_ml_results.json", "LogisticRegression"),
        "TextCNN": ("deep_textcnn_results.json", None),
        "BiLSTM": ("deep_bilstm_results.json", None),
        "RoBERTa": ("bert_roberta_results.json", None),
        "MacBERT": ("bert_macbert_results.json", None),
    }
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_cn = ["准确率", "精确率", "召回率", "F1"]
    data = {}
    for m, (fn, key) in files.items():
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        if key:  # 传统ML
            d = d[key]
        else:
            d = d["test"]
        data[m] = [d[k] for k in metrics]

    models = [m for m in MODEL_ORDER if m in data]
    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        ax.bar(x + i * width, data[m], width,
               label=MODEL_CN.get(m, m), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_cn)
    ax.set_ylabel("分数")
    ax.set_ylim(0.95, 1.005)
    ax.set_title("各模型在原始测试集上的分类性能对比")
    ax.legend(ncol=len(models), loc="lower center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "baseline_performance.png"))
    plt.close()
    print("saved baseline_performance.png")


# ========== 2-4. 鲁棒性相关图 ==========
def plot_robustness():
    d = json.load(open(os.path.join(RES, "robustness_results.json"), encoding="utf-8"))
    res = d["results"]
    induce = d["inducement_strategies"]
    evade = d["evasion_strategies"]
    strats = induce + evade
    models = [m for m in MODEL_ORDER if m in res]

    # --- 2. 检出率热力图 ---
    mat = np.array([[res[m][s]["detection_rate"] for s in ["original"] + strats]
                    for m in models])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.85, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(["original"] + strats)))
    ax.set_xticklabels([STRAT_CN[s] for s in ["original"] + strats], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_CN.get(m, m) for m in models])
    for i in range(len(models)):
        for j in range(len(["original"] + strats)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.axvline(0.5, color="black", lw=1)
    ax.axvline(len(induce) + 0.5, color="blue", lw=2, ls="--")
    ax.set_title("各模型在不同改写策略下的欺诈检出率热力图\n(蓝色虚线左:诱导增强组  右:对抗规避组)")
    fig.colorbar(im, ax=ax, label="检出率")
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "robustness_heatmap.png"))
    plt.close()
    print("saved robustness_heatmap.png")

    # --- 3. 检出率下降(ASR)分组柱状图 ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(strats))
    width = 0.16
    for i, m in enumerate(models):
        base = res[m]["original"]["detection_rate"]
        drops = [max(0, base - res[m][s]["detection_rate"]) * 100 for s in strats]
        # bootstrap 95% CI 误差棒(若结果含 delta_dr_ci95)
        yerr = None
        if all("delta_dr_ci95" in res[m][s] for s in strats):
            lo = [max(0, (base - res[m][s]["detection_rate"]) -
                      (res[m][s]["delta_dr_ci95"][0])) * 100 for s in strats]
            hi = [((res[m][s]["delta_dr_ci95"][1]) -
                   (base - res[m][s]["detection_rate"])) * 100 for s in strats]
            yerr = [np.clip(lo, 0, None), np.clip(hi, 0, None)]
        ax.bar(x + i * width, drops, width, label=MODEL_CN.get(m, m),
               color=PALETTE[i % len(PALETTE)], yerr=yerr,
               error_kw={"elinewidth": 0.7, "capsize": 1.5, "alpha": 0.6})
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([STRAT_CN[s] for s in strats], rotation=20, ha="right")
    ax.set_ylabel("检出率下降 (百分点)")
    ax.set_title("不同诱导/规避策略导致的欺诈检出率下降(攻击成功率)")
    ax.axvline(len(induce) - 0.5 + width * (len(models)) / 2, color="gray", ls="--")
    ax.text(len(induce) / 2 - 0.5, ax.get_ylim()[1] * 0.9, "诱导增强组",
            ha="center", fontsize=11, color="#555")
    ax.text(len(induce) + len(evade) / 2 - 0.5, ax.get_ylim()[1] * 0.9, "对抗规避组",
            ha="center", fontsize=11, color="#555")
    ax.legend(ncol=len(models), fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "detection_drop.png"))
    plt.close()
    print("saved detection_drop.png")

    # --- 4. 置信度变化(折线)---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = ["original"] + strats
    for i, m in enumerate(models):
        ys = [res[m][s]["mean_fraud_prob"] for s in xs]
        ax.plot(range(len(xs)), ys, marker="o", label=MODEL_CN.get(m, m),
                color=PALETTE[i % len(PALETTE)], lw=2)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([STRAT_CN[s] for s in xs], rotation=20, ha="right")
    ax.set_ylabel("平均欺诈置信度")
    ax.axvline(len(induce) + 0.5, color="blue", lw=1.5, ls="--")
    ax.set_title("各模型对欺诈样本的平均预测置信度随改写策略的变化")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "confidence_change.png"))
    plt.close()
    print("saved confidence_change.png")


# ========== 5. 混淆矩阵 ==========
def plot_confusion():
    files = {
        "RoBERTa": ("bert_roberta_results.json", None),
        "MacBERT": ("bert_macbert_results.json", None),
        "TF-IDF+LR": ("traditional_ml_results.json", "LogisticRegression"),
    }
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 3.6))
    for ax, (name, (fn, key)) in zip(axes, files.items()):
        d = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
        cm = np.array((d[key] if key else d["test"])["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["正常", "欺诈"]); ax.set_yticklabels(["正常", "欺诈"])
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title(name)
    plt.suptitle("代表性模型在原始测试集上的混淆矩阵")
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "confusion_matrices.png"))
    plt.close()
    print("saved confusion_matrices.png")


# ========== 6. 捷径诊断:关键词覆盖 + 关键词规则 vs 模型 ==========
def plot_shortcut():
    path = os.path.join(RES, "shortcut_diagnosis.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path, encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # 左:各关键词在欺诈/正常中的覆盖率
    cov = d["per_keyword_coverage"]
    kws = list(cov.keys())
    fraud_r = [cov[k]["fraud"] * 100 for k in kws]
    normal_r = [cov[k]["normal"] * 100 for k in kws]
    y = np.arange(len(kws))
    ax1.barh(y - 0.2, fraud_r, 0.4, label="欺诈", color="#C44E52")
    ax1.barh(y + 0.2, normal_r, 0.4, label="正常", color="#4C72B0")
    ax1.set_yticks(y); ax1.set_yticklabels(kws)
    ax1.invert_yaxis()
    ax1.set_xlabel("出现比例 (%)")
    ax1.set_title("欺诈关键词在两类通话中的覆盖率")
    ax1.legend()

    # 右:关键词规则 F1 vs 各模型 F1
    kr = d["keyword_rule"]
    names = ["仅关键词\n规则", "TF-IDF+LR", "TextCNN", "BiLSTM", "RoBERTa", "MacBERT"]
    f1s = [kr["f1"]]
    for fn, key in [("traditional_ml_results.json", "LogisticRegression"),
                    ("deep_textcnn_results.json", None), ("deep_bilstm_results.json", None),
                    ("bert_roberta_results.json", None), ("bert_macbert_results.json", None)]:
        try:
            dd = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
            f1s.append((dd[key] if key else dd["test"])["f1"])
        except Exception:
            f1s.append(0)
    colors = ["#937860"] + ["#55A868"] * 5
    bars = ax2.bar(names, f1s, color=colors)
    ax2.set_ylabel("F1 分数")
    ax2.set_ylim(0.6, 1.02)
    ax2.set_title("单一关键词规则 vs 学习模型(测试集 F1)")
    for b, v in zip(bars, f1s):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=8)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "shortcut_diagnosis.png"))
    plt.close()
    print("saved shortcut_diagnosis.png")


# ========== 7. 错误案例:被规避攻击翻转的样本 ==========
def plot_error_cases():
    path = os.path.join(RES, "error_cases.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path, encoding="utf-8"))
    cases = d["cases"]
    if not cases:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左:每个翻转样本的置信度 改写前->后
    pb = [c["prob_before"] for c in cases]
    pa = [c["prob_after"] for c in cases]
    order = np.argsort(pa)
    idx = np.arange(len(cases))
    for k, o in enumerate(order):
        ax1.plot([0, 1], [pb[o], pa[o]], color="#C44E52", alpha=0.35, lw=0.8)
    ax1.scatter([0] * len(cases), pb, color="#4C72B0", s=12, label="改写前", zorder=3)
    ax1.scatter([1] * len(cases), pa, color="#C44E52", s=12, label="规避后", zorder=3)
    ax1.axhline(0.5, color="gray", ls="--", lw=1)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["改写前", "规避后"])
    ax1.set_ylabel("欺诈置信度")
    ax1.set_title(f"被关键词规避翻转的 {d['n_flipped']} 个样本(BiLSTM)\n"
                  f"置信度 {d['avg_prob_before']:.2f}→{d['avg_prob_after']:.2f}")
    ax1.legend()

    # 右:翻转样本关键词数 改写前->后
    kb = [c["kw_before"] for c in cases]
    ka = [c["kw_after"] for c in cases]
    ax2.hist([kb, ka], bins=range(0, max(kb + ka) + 2), label=["改写前", "规避后"],
             color=["#4C72B0", "#C44E52"])
    ax2.set_xlabel("欺诈关键词数")
    ax2.set_ylabel("样本数")
    ax2.set_title(f"翻转样本的关键词数变化\n均值 {d['avg_kw_before']:.1f}→{d['avg_kw_after']:.1f}")
    ax2.legend()
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "error_cases.png"))
    plt.close()
    print("saved error_cases.png")


if __name__ == "__main__":
    plot_baseline_performance()
    plot_robustness()
    plot_confusion()
    plot_shortcut()
    plot_error_cases()
    print("\n所有图表已生成至", FIG)
