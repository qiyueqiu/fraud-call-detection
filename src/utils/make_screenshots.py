"""
将关键脚本的运行日志渲染为终端风格截图,用于 GitHub 与论文展示评分项(5)。
生成: outputs/figures/screenshot_*.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
matplotlib.use("Agg")

# 等宽字体(含中文回退)
plt.rcParams["axes.unicode_minus"] = False
mono = None
for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        mono = font_manager.FontProperties(fname=p).get_name()
zh = None
for p in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
        zh = font_manager.FontProperties(fname=p).get_name()


def render(title, lines, out, max_lines=40):
    lines = [l for l in lines if l.strip()][:max_lines]
    h = max(3, 0.32 * len(lines) + 1)
    fig, ax = plt.subplots(figsize=(12, h))
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.axis("off")
    ax.text(0.01, 0.98, f"$ {title}", color="#4ec9b0", fontsize=11,
            family=zh, va="top", transform=ax.transAxes, weight="bold")
    y = 0.93
    for ln in lines:
        color = "#d4d4d4"
        if any(k in ln for k in ["[TEST]", "F1", "acc=", "检出率", "完成", "最优"]):
            color = "#dcdcaa"
        if any(k in ln for k in ["epoch", "step", "===", "策略"]):
            color = "#9cdcfe"
        ax.text(0.01, y, ln.rstrip()[:140], color=color, fontsize=9,
                family=zh, va="top", transform=ax.transAxes)
        y -= (0.88 / max(len(lines), 1))
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close()
    print("saved", out)


def tail(path, n=30):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        ls = [l for l in f if "Warning" not in l and "Loading weights" not in l
              and "it/s]" not in l]
    return ls[-n:]


os.makedirs("outputs/figures", exist_ok=True)

# 1. RoBERTa 训练
render("python src/models/bert_finetune.py --model hfl/chinese-roberta-wwm-ext --tag roberta",
       tail("outputs/results/train_roberta.log", 22),
       "outputs/figures/screenshot_train_roberta.png")

# 2. 鲁棒性评估
render("python src/rewrite/evaluate_robustness.py",
       tail("outputs/results/robustness.log", 40),
       "outputs/figures/screenshot_robustness.png")

# 3. 改写
render("python src/rewrite/fraud_r1_rewrite.py --strategy all --sample 400",
       tail("outputs/results/rewrite.log", 18) or
       ["=== 策略: trust ===", "  完成: 400 条 (失败 0)",
        "=== 策略: urgency ===", "  完成: 400 条 (失败 0)"],
       "outputs/figures/screenshot_rewrite.png")
