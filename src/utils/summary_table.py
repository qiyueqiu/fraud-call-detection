"""
生成"实验结果汇总"图(用于 GitHub README 与论文附录的运行结果展示)
汇总所有模型在原始测试集与对抗规避下的关键指标到一张表格图。
"""
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, _os.path.dirname(__file__))
from figio import savefig_dual
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
matplotlib.use("Agg")

for p in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
        plt.rcParams["axes.unicode_minus"] = False

RES = "outputs/results"
rob = json.load(open(os.path.join(RES, "robustness_results.json"), encoding="utf-8"))["results"]

models = ["TF-IDF+LogisticRegression", "TextCNN", "BiLSTM", "RoBERTa", "MacBERT"]
mshow = ["TF-IDF+LR", "TextCNN", "BiLSTM", "RoBERTa", "MacBERT"]
cols = ["原始DR", "规避DR", "下降(pp)", "良性伪装DR", "谐音DR", "原始置信度", "规避置信度"]

rows = []
for m in models:
    r = rob[m]
    orig = r["original"]["detection_rate"] * 100
    ev = r["evasion"]["detection_rate"] * 100
    rows.append([
        f"{orig:.1f}", f"{ev:.1f}", f"{orig-ev:+.1f}",
        f"{r['benign_disguise']['detection_rate']*100:.1f}",
        f"{r['code_switch']['detection_rate']*100:.1f}",
        f"{r['original']['mean_fraud_prob']:.3f}",
        f"{r['evasion']['mean_fraud_prob']:.3f}",
    ])

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.axis("off")
table = ax.table(cellText=rows, rowLabels=mshow, colLabels=cols,
                 cellLoc="center", rowLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)
for (r, c), cell in table.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", weight="bold")
ax.set_title("欺诈通话检测鲁棒性实验结果汇总(检出率 DR / 平均置信度,%)",
             fontsize=12, weight="bold", pad=14)
plt.tight_layout()
savefig_dual("outputs/figures/results_summary_table.png", bbox_inches="tight")
print("saved results_summary_table.png")
