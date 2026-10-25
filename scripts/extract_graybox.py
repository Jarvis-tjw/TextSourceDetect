# -*- coding: utf-8 -*-
"""
灰盒特征提取（extract_graybox.py）
==================================
用【参考语言模型】计算每段文本的 token 级可预测性特征，
用来补充神经分类器在跨域场景下的排序能力。

为什么参考模型不能随便选：
  1) 必须是多语言模型（数据含 en / ru / ar / zh）；
  2) 不能是数据集里的生成器（bloomz / flan-t5 / davinci ...），
     否则学到的是"这段是 bloomz 写的"，而不是"这段是 AI 写的"，换生成器就失效；
  3) 版本必须固定，赛题要求可复现。
默认用 facebook/xglm-564M：多语言、不在生成器名单、8GB 显存可跑。

输出特征（每段文本一行）：
  n_tokens, mean_surprisal, std_surprisal, p10_surprisal, p90_surprisal,
  ppl, mean_entropy, std_entropy, easy_1, easy_2
  surprisal = -log p(真实 token)；ppl = exp(mean_surprisal)；
  easy_1 / easy_2 是 surprisal < 1 bit / < 2 bit 的 token 占比（GLTR 思路的简化）。

用法：
  python scripts/extract_graybox.py ^
      --input data/processed/ood/graybox_smoke.jsonl ^
      --out data/processed/ood/graybox_smoke_feat.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description="参考语言模型的 token 级特征提取")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/xglm-564M")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2, help="显存有限，默认 2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过输出文件里已有的 id")
    args = ap.parse_args()
    return args


def summary_from_tensors(surprisal, entropy, n):
    """把一条样本的逐 token 值聚合成定长特征。"""
    s = surprisal.detach().float().cpu().numpy()
    e = entropy.detach().float().cpu().numpy()
    if n <= 0 or s.size == 0:
        return None
    return {
        "n_tokens": int(n),
        "mean_surprisal": float(s.mean()),
        "std_surprisal": float(s.std()),
        "p10_surprisal": float(np.percentile(s, 10)),
        "p90_surprisal": float(np.percentile(s, 90)),
        "ppl": float(np.exp(min(s.mean(), 20.0))),
        "mean_entropy": float(e.mean()),
        "std_entropy": float(e.std()),
        "easy_1": float((s < 1.0).mean()),
        "easy_2": float((s < 2.0).mean()),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device, "| 参考模型:", args.model)

    rows = load_jsonl(args.input, args.limit)
    # 预处理产物不一定带 id（例如 indomain_val.jsonl），用稳定行号兜底，
    # 保证“特征行”和“原文件行”能一一对应。
    for i, r in enumerate(rows):
        if r.get("id") is None:
            r["id"] = i
    print("输入样本:", len(rows))

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
        rows = [r for r in rows if r["id"] not in done_ids]
        print("断点续跑，剩余:", len(rows))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).eval().to(device)

    mode = "a" if args.resume else "w"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n_done = 0

    with open(args.out, mode, encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch):
            chunk = rows[start:start + args.batch]
            texts = [r["text"] for r in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_len)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                # 用前一个位置预测下一个 token
                shift_logits = logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                mask = attn[:, 1:].bool()
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                true_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                surprisal = -true_lp
                entropy = -(log_probs.exp() * log_probs).sum(-1)
                del log_probs, shift_logits, logits

            for i, r in enumerate(chunk):
                m = mask[i]
                feat = summary_from_tensors(surprisal[i][m], entropy[i][m], int(m.sum()))
                if feat is None:
                    continue
                rec = {
                    "id": r["id"],
                    "label": r.get("label"),
                    "domain": r.get("domain"),
                    "generator": r.get("generator"),
                    "lang": r.get("lang"),
                }
                rec.update(feat)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1

            if (start // args.batch + 1) % 20 == 0:
                print("  已处理 %d / %d | 用时 %.0fs"
                      % (min(start + args.batch, len(rows)), len(rows), time.time() - t0))

    print("完成:", n_done, "条 | 总用时 %.1f 分钟" % ((time.time() - t0) / 60))
    print("特征已写出:", args.out)


if __name__ == "__main__":
    main()
