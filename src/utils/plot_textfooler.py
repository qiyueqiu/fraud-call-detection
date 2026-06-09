"""
TextFooler 对抗攻击结果可视化
================================
基于 outputs/results/textfooler_results.json 生成:
  1. 各模型攻击成功率(ASR)与攻击后准确率对比柱状图
  2. ASR / 扰动率 / 平均查询次数 三指标综合对比(揭示鲁棒性与攻击代价)
输出至 outputs/figures/{textfooler_asr,textfooler_metrics}.{pdf,png}
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
ORDER = ["TF-IDF+LR", "TextCNN", "BiLSTM", "RoBERTa", "MacBERT"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def main():
    d = json.load(open(os.path.join(RES, "textfooler_results.json"), encoding="utf-8"))
    res = d["results"]
    models = [m for m in ORDER if m in res]

    # ---- 1. ASR 与攻击后准确率 ----
    asr = [res[m]["attack_success_rate"] * 100 for m in models]
    auacc = [res[m]["accuracy_under_attack"] * 100 for m in models]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - width / 2, asr, width, label="攻击成功率 ASR", color="#C44E52")
    b2 = ax.bar(x + width / 2, auacc, width, label="攻击后准确率", color="#4C72B0")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("TextFooler 词级对抗攻击:各模型攻击成功率与攻击后准确率")
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "textfooler_asr.png"))
    plt.close()
    print("saved textfooler_asr.png")

    # ---- 2. 三指标综合(ASR 柱 + 扰动率/查询次数 折线双轴) ----
    pert = [res[m]["perturbation_rate"] * 100 for m in models]
    queries = [res[m]["avg_queries"] for m in models]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    bars = ax1.bar(x, asr, 0.5, color="#C44E52", alpha=0.75, label="ASR (%)")
    for b, v in zip(bars, asr):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_ylabel("攻击成功率 ASR (%)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x); ax1.set_xticklabels(models)

    ax2 = ax1.twinx()
    ax2.plot(x, pert, "o-", color="#55A868", lw=2, label="扰动率 (%)")
    ax2.plot(x, np.array(queries) / max(queries) * max(pert) * 1.2, "s--",
             color="#8172B3", lw=2, label="平均查询(归一化)")
    for xi, p, q in zip(x, pert, queries):
        ax2.annotate(f"{p:.1f}%", (xi, p), textcoords="offset points",
                     xytext=(0, 8), fontsize=7.5, color="#3a7d4f", ha="center")
        ax2.annotate(f"{q:.0f}", (xi, q / max(queries) * max(pert) * 1.2),
                     textcoords="offset points", xytext=(0, -12), fontsize=7.5,
                     color="#5d4f8c", ha="center")
    ax2.set_ylabel("扰动率 (%)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title("TextFooler 攻击代价对比:ASR、扰动率与平均查询次数")
    plt.tight_layout()
    savefig_dual(os.path.join(FIG, "textfooler_metrics.png"))
    plt.close()
    print("saved textfooler_metrics.png")


if __name__ == "__main__":
    main()
