# -*- coding: utf-8 -*-
"""
评测与提交工具（eval.py）
========================
统一实现赛题的评分口径，并负责生成提交文件。所有模型都用这里的函数评估，
保证口径一致、可复现。

赛题评分（初赛）：
    模型性能 = 宏平均 F1 × 70% + 短文本(<100 词)子集 F1 × 30%
  - 宏平均 F1：对 human(0)、machine(1) 两类分别算 F1 再取平均（不受类别不平衡影响，
    这正是为什么不能只看准确率）。
  - 短文本子集：词数 < 100 的样本单独算一次宏平均 F1。
  - 二分类要输出"概率"，最终提交时用固定阈值（默认 0.5）把概率转成 0/1。

关键点：阈值会明显影响宏 F1，因此提供"在本地 holdout 上搜索最优阈值"的功能。
注意：只能在本地的 holdout 上调阈值，绝不能拿官方 val 调参。

用法：
  # 评估一个预测文件（jsonl，每行 {"id","prob","label","text"}）
  python scripts/eval.py --pred data/processed/baseline_pred_holdout.jsonl
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def macro_f1(labels, preds):
    """宏平均 F1：两类 F1 的平均。"""
    return f1_score(labels, preds, average="macro", zero_division=0)


def evaluate(labels, probs, texts, threshold=0.5):
    """按赛题口径评估。
    参数：
      labels : 真实标签列表（0=human,1=machine）
      probs  : 预测"是机器"的概率列表
      texts  : 对应文本（用于判断是否 <100 词）
      threshold : 把概率转 0/1 的阈值
    返回：包含整体宏F1、短文本宏F1、加权总分等的字典。
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    overall = macro_f1(labels, preds)
    # 短文本掩码：词数 < 100
    mask_short = np.array([C.count_words(t) < 100 for t in texts])
    if mask_short.any():
        short = macro_f1(labels[mask_short], preds[mask_short])
    else:
        short = overall
    combined = 0.7 * overall + 0.3 * short  # 按官方 70%/30% 权重
    return {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "macro_f1": float(overall),
        "short_macro_f1": float(short),
        "combined": float(combined),
        "short_ratio": float(mask_short.mean()),
    }


def search_threshold(labels, probs, texts, lo=0.05, hi=0.95, steps=91):
    """在本地 holdout 上网格搜索"综合分最高"的阈值。
    返回 (best_threshold, best_metrics)。搜的是 maximize combined 分。
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])

    best = (0.5, -1.0, None)
    for th in np.linspace(lo, hi, steps):
        preds = (probs >= th).astype(int)
        overall = macro_f1(labels, preds)
        short = macro_f1(labels[mask_short], preds[mask_short]) if mask_short.any() else overall
        combined = 0.7 * overall + 0.3 * short
        if combined > best[1]:
            best = (float(th), float(combined), {
                "macro_f1": float(overall),
                "short_macro_f1": float(short),
                "combined": float(combined),
                "threshold": float(th),
            })
    return best[0], best[2]


def load_predictions(path):
    """读取预测 jsonl 文件，返回 (ids, labels, probs, texts)。
    label/text 可能不存在（官方提交时没有标签），此时对应返回 None。
    """
    ids, labels, probs, texts = [], [], [], []
    has_label = has_text = True
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids.append(r.get("id", ""))
            probs.append(float(r.get("prob", 0.5)))
            if "label" in r:
                labels.append(int(r["label"]))
            else:
                has_label = False
            if "text" in r:
                texts.append(r.get("text", ""))
            else:
                has_text = False
    return ids, (labels if has_label else None), probs, (texts if has_text else None)


def write_result_csv(ids, probs, path):
    """写提交文件 result.csv：两列 id,prob（prob 为"是机器"的概率）。
    注意：官方 val.jsonl 尚未提供，最终列名/编号规则需以其要求为准，届时按需微调。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prob"])
        for i, p in zip(ids, probs):
            w.writerow([i, "%.6f" % p])


def main():
    ap = argparse.ArgumentParser(description="评估预测文件（有标签时）并可选导出 result.csv")
    ap.add_argument("--pred", required=True, help="预测 jsonl 文件")
    ap.add_argument("--result-csv", default="", help="若提供，则导出提交用 result.csv")
    ap.add_argument("--threshold", default="auto", help="阈值：'auto' 或具体数值")
    args = ap.parse_args()

    ids, labels, probs, texts = load_predictions(args.pred)
    print("预测条数:", len(probs))

    if labels is not None and texts is not None:
        if args.threshold == "auto":
            th, m = search_threshold(labels, probs, texts)
            print("最优阈值: %.2f -> %s" % (th, json.dumps(m, ensure_ascii=False)))
        else:
            th = float(args.threshold)
            print("阈值 %.2f -> %s" % (th, json.dumps(evaluate(labels, probs, texts, th), ensure_ascii=False)))

    if args.result_csv:
        write_result_csv(ids, probs, args.result_csv)
        print("已写出:", args.result_csv)


if __name__ == "__main__":
    main()
