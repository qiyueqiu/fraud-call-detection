"""
数据预处理脚本
- 加载原始训练集/测试集
- 清洗对话文本(去除"音频内容："前缀、结尾"**"噪声、规范化空白)
- 处理 is_fraud 缺失值(根据 fraud_type 推断,无法推断则丢弃)
- 划分训练集/验证集
- 输出到 data/processed/
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "fraud_telecom_dataset"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEED = 42


def clean_dialogue(text: str) -> str:
    """清洗单条对话文本。"""
    if not isinstance(text, str):
        return ""
    # 去掉开头的 "音频内容：" 标记
    text = re.sub(r"^\s*音频内容\s*[:：]\s*", "", text)
    # 去掉结尾连续的 * 和空白
    text = re.sub(r"[\*\s]+$", "", text)
    # 规范化连续空行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def infer_is_fraud(row):
    """根据 fraud_type 补全 is_fraud 缺失值。
    有明确 fraud_type -> 欺诈;若 is_fraud 缺失且无 fraud_type 则无法判断。"""
    if pd.notna(row["is_fraud"]):
        return row["is_fraud"]
    if pd.notna(row["fraud_type"]) and str(row["fraud_type"]).strip():
        return True
    return None  # 无法推断,后续丢弃


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 统一列名(去除可能的 BOM)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["text"] = df["specific_dialogue_content"].apply(clean_dialogue)
    # 处理 is_fraud:原始可能是字符串 True/False
    df["is_fraud"] = df["is_fraud"].map(
        {True: True, False: False, "True": True, "False": False}
    ).where(df["is_fraud"].notna(), other=pd.NA)
    df["is_fraud"] = df.apply(infer_is_fraud, axis=1)
    before = len(df)
    df = df[df["is_fraud"].notna()].copy()
    df["label"] = df["is_fraud"].astype(int)
    # 去掉清洗后为空或过短的样本
    df = df[df["text"].str.len() >= 10].copy()
    after = len(df)
    print(f"  {path}: {before} -> {after} (丢弃 {before-after} 条无法判定/过短)")
    return df.reset_index(drop=True)


def main():
    print("加载并清洗数据...")
    train_full = load_and_clean(os.path.join(RAW_DIR, "训练集结果.csv"))
    test = load_and_clean(os.path.join(RAW_DIR, "测试集结果.csv"))

    # 从训练集中划分验证集(分层)
    train, val = train_test_split(
        train_full,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=train_full["label"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    cols = ["text", "label", "is_fraud", "interaction_strategy", "call_type", "fraud_type"]
    train[cols].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test[cols].to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("\n=== 数据集规模 ===")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        pos = int((d["label"] == 1).sum())
        neg = int((d["label"] == 0).sum())
        print(f"  {name}: {len(d)} (欺诈 {pos} / 正常 {neg})")
    print("\n处理完成,输出至", OUT_DIR)


if __name__ == "__main__":
    main()
