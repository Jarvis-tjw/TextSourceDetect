# -*- coding: utf-8 -*-
"""
数据预处理脚本（preprocess.py）
==============================
把原始 data.jsonl 转换成"统一、干净、可训练"的两份 jsonl：
  - data/processed/train.jsonl    本地训练集
  - data/processed/holdout.jsonl  本地验证集（用于调阈值/评估，绝不能拿官方 val 调参）

每条样本统一成这种格式：
  {"text": 文本, "label": 0/1, "generator": 生成器, "domain": 领域,
   "lang": 语言, "group": 分组键, "source_id": 源文章id}
  label：0 = human（人类撰写），1 = machine（机器生成）

遵循指南的预处理规则：
  1) 一条原始记录展开成两个样本：human 与 machine；
  2) human 按 (domain, source_id) 去重（同一篇文章只保留一个人类样本）；
  3) 过滤空文本；
  4) 过滤过短文本（默认 < 20 字符，避免噪声）。

关于切分：
  按 group 分组切分（group = domain::source_id）。
  同一篇文章产生的所有样本只进 train 或只进 holdout，避免"同源泄漏"，
  也更接近"跨领域泛化"的真实评测场景。

运行：
  python scripts/preprocess.py --holdout-ratio 0.1
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def parse_args():
    p = argparse.ArgumentParser(description="把 data.jsonl 预处理成统一的 train/holdout")
    p.add_argument("--data", default=C.DATA_PATH, help="原始 data.jsonl 路径")
    p.add_argument("--outdir", default=C.PROCESSED_DIR, help="输出目录")
    p.add_argument("--holdout-ratio", type=float, default=0.1, help="本地验证集比例")
    p.add_argument("--min-chars", type=int, default=20, help="过滤掉短于该字符数的文本")
    p.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    return p.parse_args()


def build_samples(rec, sid, group):
    """把一条原始记录拆成 human/machine 两个"半成品样本"。
    参数 sid、group 由外部算好（含 id 缺失时的兜底逻辑），保证同源样本能被正确归组。
    返回 (human_sample_or_None, machine_sample_or_None)。
    """
    domain = C.get_domain(rec)
    human = C.get_human_text(rec)
    machine = C.get_machine_text(rec)

    human_sample = None
    if human and human.strip():
        human_sample = {
            "text": human,
            "label": 0,               # 0 = human
            "generator": "human",     # 人类样本没有生成器，用占位名
            "domain": domain,
            "lang": C.detect_lang(human),
            "group": group,
            "source_id": sid,
        }

    machine_sample = None
    if machine and machine.strip():
        machine_sample = {
            "text": machine,
            "label": 1,               # 1 = machine
            "generator": C.get_generator(rec),
            "domain": domain,
            "lang": C.detect_lang(machine),
            "group": group,
            "source_id": sid,
        }
    return human_sample, machine_sample


def main():
    args = parse_args()
    random.seed(args.seed)                 # 固定随机种子，保证切分可复现
    os.makedirs(args.outdir, exist_ok=True)

    samples = []                          # 全部样本（备用）
    seen_human = set()                    # human 去重用
    group_bucket = defaultdict(list)      # 按 group 归拢样本，便于分组切分

    with open(args.data, encoding="utf-8") as f:
        for row_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sid = C.get_source_id(rec) or ("__row%d" % row_no)  # id 缺失时行号兜底
            group = "%s::%s" % (C.get_domain(rec), sid)

            human_sample, machine_sample = build_samples(rec, sid, group)

            # human 去重：同一 group 只保留第一个人类样本
            if human_sample is not None and group not in seen_human:
                seen_human.add(group)
                if len(human_sample["text"]) >= args.min_chars:
                    group_bucket[group].append(human_sample)

            if machine_sample is not None and len(machine_sample["text"]) >= args.min_chars:
                group_bucket[group].append(machine_sample)

    # -------------------- 按 group 切分 train / holdout --------------------
    groups = list(group_bucket.keys())
    random.shuffle(groups)                       # 打乱后按比例切
    n_holdout = int(len(groups) * args.holdout_ratio)
    holdout_groups = set(groups[:n_holdout])

    train_rows, holdout_rows = [], []
    for g, rows in group_bucket.items():
        (holdout_rows if g in holdout_groups else train_rows).extend(rows)
    random.shuffle(train_rows)
    random.shuffle(holdout_rows)

    # -------------------- 写文件 --------------------
    def dump(rows, name):
        path = os.path.join(args.outdir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    p_train = dump(train_rows, "train.jsonl")
    p_hold = dump(holdout_rows, "holdout.jsonl")

    # -------------------- 统计输出 --------------------
    def summarize(rows, name):
        lab = Counter(r["label"] for r in rows)
        dom = Counter(r["domain"] for r in rows)
        short_ratio = 100.0 * sum(1 for r in rows if C.count_words(r["text"]) < 100) / max(1, len(rows))
        print("[%s] 样本=%d | human=%d machine=%d | 领域数=%d | 短文本(<100词)=%.1f%%"
              % (name, len(rows), lab.get(0, 0), lab.get(1, 0), len(dom), short_ratio))

    print("预处理完成：")
    print("  写入:", p_train)
    print("  写入:", p_hold)
    summarize(train_rows, "train")
    summarize(holdout_rows, "holdout")


if __name__ == "__main__":
    main()
