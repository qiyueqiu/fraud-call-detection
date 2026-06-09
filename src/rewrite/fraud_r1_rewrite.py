"""
Fraud-R1 风格诱导改写模块
============================
参考 Fraud-R1 (2025) 的增强诱导思路,对测试集中的【欺诈】对话进行多策略语义改写,
生成对抗性测试集,用于评估已训练分类器的鲁棒性(不重新训练)。

设计的 5 种诱导策略(综合 Fraud-R1 的 D1-D3 增强级别与数据集说明的 5 个互动维度):
  1. trust       信任建立 (Building Credibility): 注入官方背书、工号、权威机构、专业术语
  2. urgency      紧迫感   (Creating Urgency):     制造时间压力、账户冻结、限时处理
  3. emotion      情感操纵 (Emotional Appeal):     利用亲情/恐惧/同情/贪婪等情绪
  4. authority    权威伪装 (Authority Disguise):   冒充公检法/银行/平台高管增强可信
  5. paraphrase   多轮语义改写 (Semantic Rewrite): 改写表述、增加铺垫轮次但保持欺诈意图

关键约束:改写【只改变表达方式与诱导强度,不改变 is_fraud 标签】。
正常通话保持原样(对照),仅对欺诈样本做诱导增强。

用法:
  python src/rewrite/fraud_r1_rewrite.py --strategy all --sample 400
"""
import os
import json
import time
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

PROC = "data/processed"
OUT = "data/rewritten"
os.makedirs(OUT, exist_ok=True)

MODEL = "deepseek-chat"
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1")

# 每种策略的系统提示词(指导 LLM 如何增强诱导)
STRATEGY_PROMPTS = {
    "trust": (
        "你是一名通信安全研究员,正在构建用于训练反诈骗模型的对抗测试样本。"
        "请把下面这通【诈骗电话】对话改写得更具【可信度】:为主叫方(left)注入官方背书元素——"
        "如具体工号、部门名称、备案编号、专业术语、引用真实业务流程、礼貌规范的话术,"
        "让对话显得更像正规客服/机构。"
        "要求:严格保持其【诈骗本质和最终诈骗意图不变】(仍在诱导受害者点链接/转账/泄露信息),"
        "保持 left/right 双方对话格式,保持中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    "urgency": (
        "你是一名通信安全研究员,正在构建反诈骗模型的对抗测试样本。"
        "请把下面这通【诈骗电话】对话改写得更具【紧迫感】:加入时间压力元素——"
        "如'账户将在X分钟内冻结''今天是最后期限''逾期将影响征信''名额仅剩最后一个'等,"
        "迫使受害者立即行动、来不及思考。"
        "要求:严格保持其【诈骗本质和最终诈骗意图不变】,保持 left/right 对话格式,"
        "保持中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    "emotion": (
        "你是一名通信安全研究员,正在构建反诈骗模型的对抗测试样本。"
        "请把下面这通【诈骗电话】对话改写得更具【情感操纵】色彩:利用受害者的情绪——"
        "如亲情(家人出事)、恐惧(涉嫌犯罪)、同情、贪婪(巨额回报)、信任(老熟人)等,"
        "通过情绪渲染降低受害者的警惕。"
        "要求:严格保持其【诈骗本质和最终诈骗意图不变】,保持 left/right 对话格式,"
        "保持中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    "authority": (
        "你是一名通信安全研究员,正在构建反诈骗模型的对抗测试样本。"
        "请把下面这通【诈骗电话】对话改写为【权威伪装】版本:让主叫方冒充更高权威——"
        "如公安局/检察院/法院、银行风控中心、电信运营商安全部门、知名平台官方等,"
        "使用威严、专业、不容置疑的口吻增强压迫感和可信度。"
        "要求:严格保持其【诈骗本质和最终诈骗意图不变】,保持 left/right 对话格式,"
        "保持中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    "paraphrase": (
        "你是一名通信安全研究员,正在构建反诈骗模型的对抗测试样本。"
        "请把下面这通【诈骗电话】对话做【多轮语义改写】:改变用词和句式表达,"
        "适当增加 1-2 轮寒暄铺垫,让对话更自然、更口语化、更难通过关键词识别,"
        "但【整体诈骗剧本和最终意图完全不变】。"
        "要求:保持 left/right 对话格式,保持中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    # ===== 对抗规避策略(主动隐藏欺诈关键词,模拟攻击者绕过检测器)=====
    "evasion": (
        "你是一名反诈对抗研究员,正在构建用于评估检测器鲁棒性的【规避型】对抗样本。"
        "请把下面这通【诈骗电话】对话改写为【隐蔽规避版本】:在【完全保留诈骗意图和剧本】的前提下,"
        "刻意【避免使用敏感词】——把'链接/点击/转账/验证码/银行卡/密码/冻结/二维码'等典型诈骗词,"
        "替换为委婉、间接、口语化的说法(如'我发个东西给你''你照着弄一下''把那串数字念给我''你那个卡的事'),"
        "用暗示和上下文让受害者理解,但表面文字不出现明显诈骗特征词。"
        "要求:保持 left/right 对话格式与中文口语风格,诈骗目的不变,只输出改写后的对话正文,不要解释。"
    ),
    "benign_disguise": (
        "你是一名反诈对抗研究员,正在构建评估检测器鲁棒性的【伪装型】对抗样本。"
        "请把下面这通【诈骗电话】对话改写得【表面上极像一通正常的客服/服务来电】:"
        "大量加入正常业务寒暄、礼貌用语、常规服务流程的描述,把诈骗诱导部分包裹、稀释在大段正常对话里,"
        "让前半段听起来完全无害,诈骗意图只在后段隐晦地、自然地引出。"
        "要求:【整体仍是诈骗(意图不变)】,保持 left/right 对话格式与中文口语风格,只输出改写后的对话正文,不要解释。"
    ),
    "code_switch": (
        "你是一名反诈对抗研究员,正在构建评估检测器鲁棒性的对抗样本。"
        "请把下面这通【诈骗电话】对话改写为【含谐音、错别字、隐语的规避版本】:"
        "把关键诈骗词用谐音字、拼音首字母、拆字或网络黑话替代(如'转账'→'专帐/zz','验证码'→'盐证马/那串号',"
        "'银行卡'→'银航咔','链接'→'连界/那个网址'),模拟诈骗分子规避关键词过滤的真实手法。"
        "要求:【诈骗意图完全不变】,受害者仍能听懂,保持 left/right 对话格式,只输出改写后的对话正文,不要解释。"
    ),
}


def rewrite_one(text, strategy, max_retries=3):
    sys_prompt = STRATEGY_PROMPTS[strategy]
    user_prompt = f"待改写的对话:\n{text}"
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.9,
            )
            content = r.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [失败] {strategy}: {str(e)[:60]}")
            time.sleep(2 ** attempt)
    return None  # 失败则返回 None,后续用原文兜底


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all",
                    help="trust/urgency/emotion/authority/paraphrase/all")
    ap.add_argument("--sample", type=int, default=400,
                    help="每种策略改写的欺诈样本数(控制 API 成本)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test = pd.read_csv(os.path.join(PROC, "test.csv"))
    fraud = test[test["label"] == 1].reset_index(drop=True)
    # 固定采样,保证各策略改写的是同一批欺诈样本(可比)
    if args.sample > 0 and args.sample < len(fraud):
        fraud = fraud.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    print(f"待改写欺诈样本: {len(fraud)} 条")

    strategies = list(STRATEGY_PROMPTS.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        print(f"\n=== 策略: {strat} ===")
        results = [None] * len(fraud)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(rewrite_one, row["text"], strat): i
                    for i, row in fraud.iterrows()}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  进度 {done}/{len(fraud)}  ({time.time()-t0:.0f}s)")

        out_df = fraud.copy()
        out_df["original_text"] = out_df["text"]
        # 改写失败的用原文兜底(标记)
        n_fail = sum(1 for r in results if r is None)
        out_df["text"] = [r if r else o for r, o in zip(results, out_df["text"])]
        out_df["rewrite_success"] = [r is not None for r in results]
        out_df["strategy_applied"] = strat
        path = os.path.join(OUT, f"test_{strat}.csv")
        out_df.to_csv(path, index=False)
        print(f"  完成: {len(out_df)} 条 (失败 {n_fail}),保存至 {path} ,耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
