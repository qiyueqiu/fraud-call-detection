"""
探索性数据分析(EDA)与可视化
生成数据集统计图,用于论文实验章节。
"""
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, _os.path.dirname(__file__))
from figio import savefig_dual
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

# 中文字体设置
def setup_chinese_font():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print("使用中文字体:", name)
            return
    print("警告: 未找到中文字体,图中文可能乱码")

setup_chinese_font()

PROC = "data/processed"
FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

train = pd.read_csv(os.path.join(PROC, "train.csv"))
val = pd.read_csv(os.path.join(PROC, "val.csv"))
test = pd.read_csv(os.path.join(PROC, "test.csv"))
full = pd.concat([train, val, test], ignore_index=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]

# 图1: 标签分布(train/val/test)
fig, ax = plt.subplots(figsize=(6, 4))
splits = {"训练集": train, "验证集": val, "测试集": test}
labels = list(splits.keys())
fraud = [int((d["label"] == 1).sum()) for d in splits.values()]
normal = [int((d["label"] == 0).sum()) for d in splits.values()]
x = range(len(labels))
ax.bar(x, fraud, label="欺诈", color="#C44E52")
ax.bar(x, normal, bottom=fraud, label="正常", color="#4C72B0")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("样本数")
ax.set_title("各数据集划分的标签分布")
ax.legend()
for i, (f, n) in enumerate(zip(fraud, normal)):
    ax.text(i, f / 2, str(f), ha="center", va="center", color="white", fontsize=9)
    ax.text(i, f + n / 2, str(n), ha="center", va="center", color="white", fontsize=9)
plt.tight_layout()
savefig_dual(os.path.join(FIG, "label_distribution.png"))
plt.close()

# 图2: 对话长度分布(按是否欺诈)
fig, ax = plt.subplots(figsize=(6, 4))
full["char_len"] = full["text"].str.len()
ax.hist(full[full["label"] == 1]["char_len"], bins=40, alpha=0.6,
        label="欺诈", color="#C44E52")
ax.hist(full[full["label"] == 0]["char_len"], bins=40, alpha=0.6,
        label="正常", color="#4C72B0")
ax.set_xlabel("对话字符数")
ax.set_ylabel("频次")
ax.set_title("对话长度分布")
ax.legend()
ax.set_xlim(0, 1500)
plt.tight_layout()
savefig_dual(os.path.join(FIG, "length_distribution.png"))
plt.close()

# 图3: 诈骗类型分布
fig, ax = plt.subplots(figsize=(7, 4))
ft = full[full["fraud_type"].notna()]["fraud_type"].value_counts()
ax.barh(ft.index[::-1], ft.values[::-1], color=PALETTE[:len(ft)][::-1])
ax.set_xlabel("样本数")
ax.set_title("欺诈类型分布")
for i, v in enumerate(ft.values[::-1]):
    ax.text(v + 20, i, str(v), va="center", fontsize=9)
plt.tight_layout()
savefig_dual(os.path.join(FIG, "fraud_type_distribution.png"))
plt.close()

# 图4: 通话类型分布
fig, ax = plt.subplots(figsize=(7, 4))
ct = full[full["call_type"].notna()]["call_type"].value_counts().head(8)
ax.barh(ct.index[::-1], ct.values[::-1], color=PALETTE[2])
ax.set_xlabel("样本数")
ax.set_title("通话类型分布(Top 8)")
plt.tight_layout()
savefig_dual(os.path.join(FIG, "call_type_distribution.png"))
plt.close()

# 图5: 互动策略 vs 是否欺诈
fig, ax = plt.subplots(figsize=(7, 4))
ct2 = pd.crosstab(full["interaction_strategy"], full["label"])
ct2.columns = ["正常", "欺诈"]
ct2 = ct2.sort_values("欺诈", ascending=True)
ct2.plot(kind="barh", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_xlabel("样本数")
ax.set_ylabel("互动策略")
ax.set_title("互动策略与欺诈标签的关系")
plt.tight_layout()
savefig_dual(os.path.join(FIG, "strategy_vs_fraud.png"))
plt.close()

print("EDA 图已生成至", FIG)
print("\n=== 统计摘要 ===")
print("总样本:", len(full))
print("对话长度: 均值 %.1f, 中位数 %.0f, 最大 %d" % (
    full["char_len"].mean(), full["char_len"].median(), full["char_len"].max()))
print("欺诈占比: %.1f%%" % (100 * full["label"].mean()))
