# -*- coding: utf-8 -*-
"""
真 OOD 实验数据切分（make_ood_split.py）
=======================================
【为什么需要它】
普通的 holdout 是从同一批数据里按组随机切的，15 个领域在训练时全都见过，
所以它只能测"同分布泛化"，测不出"跨领域泛化"。

本脚本把某一个领域整体挖掉，得到真正的域外测试集：
    train_ood.jsonl    : 训练集，排除掉目标领域
    ood_test.jsonl     : 目标领域的全部样本（训练时完全没见过）
    indomain_val.jsonl : 非目标领域的 holdout 子集

【为什么用"拼接已有产物"而不是重新预处理】
preprocess.py 已经按 group 把数据切成 train / holdout，两边的 group 互不相交。
把 train 和 holdout 中属于目标领域的行合并，就得到该领域【全部】样本，
既完整又不会重复。这样不用重跑预处理，也不会引入新的切分随机性。

用法：
  python scripts/make_ood_split.py --domain arxiv
  python scripts/make_ood_split.py --domain wikipedia-20220301.en
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def stream_filter(path, keep_fn):
    """流式读取并筛选 jsonl，避免把上 GB 文件一次性读进内存。"""
    kept = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if keep_fn(r):
                kept += 1
                yield r
    # 生成器函数无法直接返回计数，这里只用于说明；统计由调用方各自完成


def main():
    ap = argparse.ArgumentParser(description="构造真 OOD 实验集")
    ap.add_argument("--domain", required=True, help="要整体留出的领域名，如 arxiv")
    ap.add_argument("--train", default=os.path.join(C.PROCESSED_DIR, "train.jsonl"))
    ap.add_argument("--holdout", default=os.path.join(C.PROCESSED_DIR, "holdout.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(C.PROCESSED_DIR, "ood"))
    ap.add_argument("--eval-frac", type=float, default=0.0,
                    help="从 ood_test 按 group 稳定抽样该比例，另存 eval_<domain>.jsonl；0=不生成")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    target = args.domain

    def is_target(r):
        return r.get("domain") == target

    # 先检查目标领域是否存在，避免拼错领域名后静默产出空测试集
    found = 0
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get("domain") == target:
                found += 1
    if found == 0:
        raise SystemExit("训练集里找不到领域 '%s'，请先用 analyze_data.py 确认领域名。" % target)
    print("目标领域 '%s' 在训练集中有 %d 条（另有一部分在 holdout 中）。" % (target, found))

    paths = {
        "train_ood": os.path.join(args.outdir, "train_ood.jsonl"),
        "ood_test": os.path.join(args.outdir, "ood_test.jsonl"),
        "indomain_val": os.path.join(args.outdir, "indomain_val.jsonl"),
    }

    counts = {"train_ood": 0, "ood_test": 0, "indomain_val": 0}
    labels = {"train_ood": [0, 0], "ood_test": [0, 0], "indomain_val": [0, 0]}

    with open(paths["train_ood"], "w", encoding="utf-8") as f_tr, \
         open(paths["ood_test"], "w", encoding="utf-8") as f_ood, \
         open(paths["indomain_val"], "w", encoding="utf-8") as f_val:

        # 训练集：排除目标领域 -> train_ood；目标领域 -> ood_test
        with open(args.train, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if is_target(r):
                    f_ood.write(line + "\n")
                    counts["ood_test"] += 1
                    labels["ood_test"][int(r["label"])] += 1
                else:
                    f_tr.write(line + "\n")
                    counts["train_ood"] += 1
                    labels["train_ood"][int(r["label"])] += 1

        # holdout：目标领域 -> ood_test（补齐剩余 10%）；其余 -> indomain_val
        with open(args.holdout, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if is_target(r):
                    f_ood.write(line + "\n")
                    counts["ood_test"] += 1
                    labels["ood_test"][int(r["label"])] += 1
                else:
                    f_val.write(line + "\n")
                    counts["indomain_val"] += 1
                    labels["indomain_val"][int(r["label"])] += 1

    for k in ("train_ood", "ood_test", "indomain_val"):
        h, m = labels[k]
        print("%-13s %6d 条 | human %6d / machine %6d | path=%s"
              % (k, counts[k], h, m, paths[k]))

    # 评估子集：OOD 检查每次都要重新推理，全量太慢；按 group 抽样保证
    # 同一源文章的人/机版本同进同出，不会出现"半篇文章在评估集里"。
    if args.eval_frac > 0:
        eval_path = os.path.join(args.outdir, "eval_%s.jsonl" % target)
        n_eval, groups, lab = 0, set(), [0, 0]
        with open(paths["ood_test"], encoding="utf-8") as f, \
             open(eval_path, "w", encoding="utf-8") as fo:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                g = str(r.get("group") or r.get("source_id") or "")
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                if (h % 10000) / 10000.0 < args.eval_frac:
                    fo.write(line + "\n")
                    n_eval += 1
                    lab[int(r["label"])] += 1
                    groups.add(g)
        print("eval 子集 %s：%d 条 / %d 组 | human %d / machine %d"
              % (eval_path, n_eval, len(groups), lab[0], lab[1]))


if __name__ == "__main__":
    main()
