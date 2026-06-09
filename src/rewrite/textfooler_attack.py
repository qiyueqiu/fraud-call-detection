"""
TextFooler 对抗攻击(中文离线适配版)
=====================================
参考 Jin et al. (2020) "Is BERT Really Robust? A Strong Baseline for Natural
Language Attack on Text Classification" (AAAI 2020) 的 TextFooler 算法,
对中文【欺诈通话二分类器】实施黑盒对抗攻击,评估其鲁棒性。

与 §3.2 的 LLM 语义改写不同,TextFooler 是【词级、自动、黑盒】的对抗攻击:
不调用大模型,仅通过对分类器的查询反馈,贪心地做最小化的同义词替换。

算法四步(完全离线,复用项目本地中文 BERT,无需 synonyms 付费词库 / TextAttack):
  1. 词重要性排序(Word Importance Ranking, WIR):jieba 分词后逐词删除,
     以目标类置信度下降量 I(w) 排序,停用词/标点不参与。
  2. 候选生成:用原始 MacBERT 作掩码语言模型(MLM),将目标词 mask 后取
     top-K 预测词作为【上下文同义候选】(BERT-Attack 式,语义更贴合中文语境)。
  3. 约束过滤:
     (a) 词性约束:候选与原词的 jieba.posseg 词性首字母一致;
     (b) 语义相似度约束:替换后整句与原句的句向量余弦 >= sim_thresh
         (句向量用被攻击的 BERT 或独立编码器;退化时用字级 Jaccard 近似)。
  4. 贪心替换:按重要性从高到低逐词替换为"使目标类概率下降最多"的候选;
     一旦预测翻转即停止,返回对抗样本。

评估指标(TextFooler 标准):
  - 攻击成功率 ASR = 翻转成功数 / 原本分类正确数
  - 扰动率 Pert% = 平均(被替换词数 / 句子词数)
  - 平均查询次数 Queries = 平均每条样本对分类器的查询次数

输出:outputs/results/textfooler_results.json
       data/rewritten/test_textfooler_<model>.csv(逐样本对抗文本)
用法:python src/rewrite/textfooler_attack.py --sample 200 --targets bilstm textcnn ...
"""
import os
import re
import sys
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForMaskedLM)

jieba.setLogLevel(20)
PROC = "data/processed"
REWRITTEN = "data/rewritten"
MODELS = "outputs/models"
RES = "outputs/results"
os.makedirs(REWRITTEN, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLM_PATH = "/home/qiqi/.cache/hf_manual/chinese-macbert-base"

# 中文停用词/标点:不参与重要性排序与替换
_PUNCT = set("，。！？、；：“”‘’（）()【】[]…—-~·\"' \n\t\r")
_STOP = set(["的", "了", "是", "我", "你", "您", "他", "她", "它", "们", "这", "那",
             "在", "和", "就", "都", "也", "要", "有", "吗", "呢", "啊", "吧",
             "left", "right", "：", ":"])


# ============ 被攻击模型的统一预测接口(返回 概率向量 [p0,p1]) ============
_cache = {}


def _load_traditional():
    if "trad" not in _cache:
        import joblib
        vec = __import__("joblib").load(os.path.join(MODELS, "tfidf_vectorizer.joblib"))
        cand = [f for f in os.listdir(MODELS) if f.startswith("best_traditional_")][0]
        clf = joblib.load(os.path.join(MODELS, cand))
        _cache["trad"] = (vec, clf)
    return _cache["trad"]


def _proba_traditional(texts):
    vec, clf = _load_traditional()
    X = vec.transform([" ".join(jieba.cut(str(t))) for t in texts])
    if hasattr(clf, "predict_proba"):
        p1 = clf.predict_proba(X)[:, 1]
    else:
        d = clf.decision_function(X)
        p1 = 1 / (1 + np.exp(-d))
    return np.stack([1 - p1, p1], axis=1)


def _load_deep(tag):
    key = f"deep_{tag}"
    if key not in _cache:
        sys.path.insert(0, "src/models")
        from deep_models import TextCNN, BiLSTMAttn
        with open(os.path.join(MODELS, f"deep_{tag}_vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        model = TextCNN(len(vocab)) if tag == "textcnn" else BiLSTMAttn(len(vocab))
        model.load_state_dict(torch.load(os.path.join(MODELS, f"deep_{tag}.pt")))
        model.to(DEVICE).eval()
        _cache[key] = (model, vocab)
    return _cache[key]


def _proba_deep(texts, tag, max_len=400):
    model, vocab = _load_deep(tag)
    from deep_models import encode
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            batch = texts[i:i + 128]
            x = torch.tensor([encode(t, vocab, max_len) for t in batch]).to(DEVICE)
            out.append(F.softmax(model(x), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _load_bert(tag):
    key = f"bert_{tag}"
    if key not in _cache:
        path = os.path.join(MODELS, f"bert_{tag}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
        _cache[key] = (model, tok)
    return _cache[key]


def _proba_bert(texts, tag, max_len=256):
    model, tok = _load_bert(tag)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = list(texts[i:i + 64])
            enc = tok(batch, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(DEVICE)
            out.append(F.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


class Victim:
    """被攻击分类器的封装,统计查询次数。"""
    def __init__(self, kind, tag):
        self.kind, self.tag = kind, tag
        self.queries = 0

    def proba(self, texts):
        self.queries += len(texts)
        if self.kind == "traditional":
            return _proba_traditional(texts)
        if self.kind == "deep":
            return _proba_deep(texts, self.tag)
        return _proba_bert(texts, self.tag)


# ============ MLM 候选生成器(原始 MacBERT,离线) ============
class MLMCandidate:
    def __init__(self, path=MLM_PATH, topk=40):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.mlm = AutoModelForMaskedLM.from_pretrained(path).to(DEVICE).eval()
        self.topk = topk
        self.mask = self.tok.mask_token

    def candidates(self, words, idx):
        """把 words[idx] 替换为 [MASK],用 MLM 预测 top-K 个同位置候选词。
        以词为单位 mask(多字词 mask 等长 [MASK]),取整词预测近似。"""
        target = words[idx]
        n_char = len(target)
        masked = words[:idx] + [self.mask * n_char] + words[idx + 1:]
        text = "".join(masked)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        ids = enc["input_ids"][0]
        mask_pos = (ids == self.tok.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            return []
        with torch.no_grad():
            logits = self.mlm(**enc).logits[0]
        # 逐 mask 位取 top 候选,拼成与原词等长的候选词
        per_pos = []
        for mp in mask_pos[:n_char]:
            topids = logits[mp].topk(self.topk).indices.cpu().numpy().tolist()
            per_pos.append([self.tok.convert_ids_to_tokens(i) for i in topids])
        cands = set()
        if n_char == 1:
            for t in per_pos[0]:
                if self._valid_char(t):
                    cands.add(t)
        else:
            # 多字:对每个 mask 位独立取 top,组合首选项 + 各位替换(控制规模)
            base = [p[0] for p in per_pos]
            for pos in range(len(per_pos)):
                for t in per_pos[pos][:12]:
                    if not self._valid_char(t):
                        continue
                    w = base.copy(); w[pos] = t
                    cand = "".join(w)
                    if self._valid_word(cand):
                        cands.add(cand)
        cands.discard(target)
        return list(cands)

    @staticmethod
    def _valid_char(t):
        return bool(t) and not t.startswith("##") and "[" not in t and \
            re.fullmatch(r"[一-鿿]", t) is not None

    @staticmethod
    def _valid_word(w):
        return re.fullmatch(r"[一-鿿]+", w) is not None


def pos_first(flag):
    return flag[0] if flag else "x"


def sent_sim_bert(orig, advs, victim_bert_tag):
    """用 RoBERTa 的 [CLS] 表示批量计算 orig 与每个 adv 的句向量余弦。
    orig: str;advs: List[str];返回 np.ndarray[len(advs)]。"""
    model, tok = _load_bert(victim_bert_tag)
    with torch.no_grad():
        enc = tok([orig] + list(advs), truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(DEVICE)
        out = model.base_model(**enc).last_hidden_state[:, 0]  # CLS
        v = F.normalize(out, dim=-1)
        sims = (v[1:] * v[0:1]).sum(dim=-1).cpu().numpy()
    return sims


def char_jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


# ============ 单样本 TextFooler 攻击 ============
def attack_one(text, victim, mlm, sim_encoder_tag, sim_thresh=0.7, max_perturb_ratio=0.4):
    """对单条文本攻击。目标:把"欺诈(类1)"翻转为"正常(类0)"。
    返回 dict(成功标志、对抗文本、扰动词数、词数、查询数增量在 victim.queries 中累计)。"""
    words = [w for w in jieba.lcut(text) if w]
    n = len(words)
    # 原始预测
    p0 = victim.proba([text])[0]
    orig_label = int(np.argmax(p0))
    if orig_label != 1:
        return {"skipped": True, "success": False, "adv_text": text,
                "n_words": n, "n_changed": 0, "orig_label": orig_label}

    # ---- 1. 词重要性排序(delete 法) ----
    deletable = [i for i, w in enumerate(words)
                 if w not in _STOP and w not in _PUNCT
                 and re.search(r"[一-鿿]", w)]
    del_texts = ["".join(words[:i] + words[i + 1:]) for i in deletable]
    importances = []
    if del_texts:
        probs = victim.proba(del_texts)  # 批量查询
        base_fraud = p0[1]
        for k, i in enumerate(deletable):
            importances.append((i, base_fraud - probs[k][1]))  # 删词后欺诈概率下降越多越重要
    importances.sort(key=lambda t: -t[1])
    order = [i for i, _ in importances]

    # 预存原词词性
    pos_map = {}
    for w in pseg.cut(text):
        pos_map.setdefault(w.word, pos_first(w.flag))

    # ---- 2-4. 贪心替换 ----
    cur_words = words.copy()
    changed = 0
    max_changes = max(1, int(max_perturb_ratio * len(deletable)))
    for i in order:
        if changed >= max_changes:
            break
        orig_w = cur_words[i]
        cands = mlm.candidates(cur_words, i)
        if not cands:
            continue
        # (a) 词性约束
        ow_pos = pos_map.get(orig_w, "x")
        cands = [c for c in cands if pos_first(
            next((w.flag for w in pseg.cut(c)), "x")) == ow_pos] or cands[:10]
        # 构造候选句并批量查询
        cand_texts = ["".join(cur_words[:i] + [c] + cur_words[i + 1:]) for c in cands]
        probs = victim.proba(cand_texts)
        # (b) 语义相似度约束(批量计算)
        if sim_encoder_tag:
            sims = sent_sim_bert(text, cand_texts, sim_encoder_tag)
        else:
            sims = np.array([char_jaccard(text, ct) for ct in cand_texts])
        best = None
        for k, (c, ct, pr) in enumerate(zip(cands, cand_texts, probs)):
            if sims[k] < sim_thresh:
                continue
            # 选使欺诈概率下降最多者
            if best is None or pr[1] < best[2]:
                best = (c, ct, pr[1], pr)
        if best is None:
            continue
        c, ct, fraud_p, pr = best
        if fraud_p < p0[1]:  # 只在确实降低欺诈概率时接受
            cur_words[i] = c
            changed += 1
            if int(np.argmax(pr)) == 0:  # 翻转成功
                return {"skipped": False, "success": True, "adv_text": ct,
                        "n_words": n, "n_changed": changed, "orig_label": 1,
                        "final_fraud_prob": float(fraud_p)}
    adv_text = "".join(cur_words)
    final_p = victim.proba([adv_text])[0]
    return {"skipped": False, "success": int(np.argmax(final_p)) == 0,
            "adv_text": adv_text, "n_words": n, "n_changed": changed,
            "orig_label": 1, "final_fraud_prob": float(final_p[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", nargs="+",
                    default=["bilstm", "textcnn", "traditional", "roberta", "macbert"])
    ap.add_argument("--sim_thresh", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max_perturb_ratio", type=float, default=0.4)
    args = ap.parse_args()

    test = pd.read_csv(os.path.join(PROC, "test.csv"))
    fraud = test[test["label"] == 1].reset_index(drop=True)
    if 0 < args.sample < len(fraud):
        fraud = fraud.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    texts = fraud["text"].tolist()
    print(f"待攻击欺诈样本: {len(texts)} 条")

    print("加载 MLM 候选生成器(原始 MacBERT)...")
    mlm = MLMCandidate(topk=args.topk)

    kind_map = {"traditional": ("traditional", None),
                "textcnn": ("deep", "textcnn"), "bilstm": ("deep", "bilstm"),
                "roberta": ("bert", "roberta"), "macbert": ("bert", "macbert")}
    name_map = {"traditional": "TF-IDF+LR", "textcnn": "TextCNN", "bilstm": "BiLSTM",
                "roberta": "RoBERTa", "macbert": "MacBERT"}
    # 语义相似度编码器:统一用 roberta 的 BERT 表示(独立于被攻击模型,公平)
    sim_tag = "roberta" if os.path.isdir(os.path.join(MODELS, "bert_roberta")) else None

    all_results = {}
    for tgt in args.targets:
        if tgt not in kind_map:
            continue
        kind, tag = kind_map[tgt]
        if kind == "deep" and not os.path.exists(os.path.join(MODELS, f"deep_{tag}.pt")):
            continue
        if kind == "bert" and not os.path.isdir(os.path.join(MODELS, f"bert_{tag}")):
            continue
        mname = name_map[tgt]
        print(f"\n=== 攻击目标: {mname} ===")
        victim = Victim(kind, tag)
        t0 = time.time()
        recs = []
        for j, txt in enumerate(texts):
            q_before = victim.queries
            r = attack_one(txt, victim, mlm, sim_tag,
                           sim_thresh=args.sim_thresh,
                           max_perturb_ratio=args.max_perturb_ratio)
            r["queries"] = victim.queries - q_before
            r["original_text"] = txt
            recs.append(r)
            if (j + 1) % 25 == 0:
                done = j + 1
                succ = sum(x["success"] for x in recs)
                skip = sum(x["skipped"] for x in recs)
                print(f"  {done}/{len(texts)}  成功 {succ}  跳过 {skip}  ({time.time()-t0:.0f}s)")

        attacked = [r for r in recs if not r["skipped"]]
        n_attacked = len(attacked)
        n_success = sum(r["success"] for r in attacked)
        succ_recs = [r for r in attacked if r["success"]]
        asr = n_success / max(1, n_attacked)
        # 扰动率:成功样本平均(改词数/词数)
        pert = float(np.mean([r["n_changed"] / max(1, r["n_words"]) for r in succ_recs])) \
            if succ_recs else 0.0
        avg_q = float(np.mean([r["queries"] for r in attacked])) if attacked else 0.0
        # 攻击后准确率(原本判对的欺诈中,攻击后仍判欺诈的比例)
        after_acc = (n_attacked - n_success) / max(1, n_attacked)

        all_results[mname] = {
            "n_total": len(texts), "n_attacked": n_attacked,
            "n_skipped": len(texts) - n_attacked, "n_success": n_success,
            "attack_success_rate": round(asr, 4),
            "perturbation_rate": round(pert, 4),
            "avg_queries": round(avg_q, 1),
            "accuracy_under_attack": round(after_acc, 4),
        }
        print(f"  [结果] ASR={asr:.4f}  扰动率={pert:.4f}  "
              f"平均查询={avg_q:.1f}  攻击后准确率={after_acc:.4f}")

        # 保存逐样本对抗文本
        df = fraud.copy()
        df["original_text"] = df["text"]
        df["adv_text"] = [r["adv_text"] for r in recs]
        df["attack_success"] = [r["success"] for r in recs]
        df["n_changed"] = [r["n_changed"] for r in recs]
        df.to_csv(os.path.join(REWRITTEN, f"test_textfooler_{tgt}.csv"), index=False)

    out = {
        "method": "TextFooler (Jin et al. 2020), 中文离线适配: jieba WIR + MacBERT-MLM 候选 + 词性/句向量约束",
        "n_samples": len(texts),
        "sim_thresh": args.sim_thresh, "mlm_topk": args.topk,
        "max_perturb_ratio": args.max_perturb_ratio,
        "metrics_def": {
            "attack_success_rate": "翻转成功数/原本分类正确数",
            "perturbation_rate": "成功样本平均(改词数/词数)",
            "avg_queries": "平均每条样本对分类器查询次数",
            "accuracy_under_attack": "攻击后欺诈样本上的准确率(越低攻击越强)",
        },
        "results": all_results,
    }
    with open(os.path.join(RES, "textfooler_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nTextFooler 结果已保存至 outputs/results/textfooler_results.json")


if __name__ == "__main__":
    main()
