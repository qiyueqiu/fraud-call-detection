"""
改写意图保留校验(任务 2 的可信度支撑)
=========================================
对抗规避/诱导改写的核心前提是:改写后【仍然是欺诈】。
若 LLM 把对话改得不再是诈骗,则检出率下降就是假象。
本脚本用 LLM 作为独立裁判,对改写样本二次判定是否仍保留欺诈意图,
报告各策略的【意图保留率】,作为鲁棒性结论的可信度支撑。

用法: python src/rewrite/verify_intent.py --strategy all --sample 100
"""
import os
import json
import time
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

REWRITTEN = "data/rewritten"
RES = "outputs/results"
os.makedirs(RES, exist_ok=True)

MODEL = "deepseek-chat"
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1")

STRATS = ["trust", "urgency", "emotion", "authority", "paraphrase",
          "evasion", "benign_disguise", "code_switch"]

JUDGE_PROMPT = (
    "你是一名反诈专家。下面是一通电话对话。请判断它【本质上是否仍是一通诈骗电话】"
    "(即主叫方是否在试图欺骗、诱导受话方转账、泄露信息、下载可疑应用或执行有害操作)。"
    "只回答一个 JSON:{\"is_fraud\": true 或 false, \"reason\": \"简短理由\"}。"
    "注意:即使话术委婉、未出现明显敏感词,只要诈骗意图存在,仍应判为 true。"
)


def judge_one(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": JUDGE_PROMPT},
                          {"role": "user", "content": f"对话:\n{text}"}],
                max_tokens=200, temperature=0.0,
                response_format={"type": "json_object"},
            )
            obj = json.loads(r.choices[0].message.content)
            return bool(obj.get("is_fraud", True))
        except Exception:
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--sample", type=int, default=100,
                    help="每策略抽样判定数(控制成本)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    strategies = STRATS if args.strategy == "all" else [args.strategy]
    summary = {}
    for strat in strategies:
        path = os.path.join(REWRITTEN, f"test_{strat}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if args.sample > 0 and args.sample < len(df):
            df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        texts = df["text"].tolist()
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        valid = [r for r in results if r is not None]
        keep = sum(1 for r in valid if r)
        rate = keep / len(valid) if valid else 0
        summary[strat] = {"n_judged": len(valid), "n_fraud": keep,
                          "intent_retention_rate": round(rate, 4)}
        print(f"{strat:15s}: 意图保留率 {rate:.3f} ({keep}/{len(valid)})")

    with open(os.path.join(RES, "intent_retention.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    overall = sum(s["intent_retention_rate"] for s in summary.values()) / len(summary)
    print(f"\n平均意图保留率: {overall:.3f}")
    print("结果已保存至 outputs/results/intent_retention.json")


if __name__ == "__main__":
    main()
