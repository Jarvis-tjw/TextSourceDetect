# -*- coding: utf-8 -*-
"""
合并 K 折预测（merge_cv.py）
============================
把 train_gpu.py 产出的 pred_fold0..K-1.jsonl 拼成完整的 OOF 预测，
按赛题口径算一次总分，并落盘给 calibrate.py 使用。

为什么 OOF 很重要：
  同一条样本只在“它没参与训练的那一折”上被打分，
  所以 OOF 分数是干净的交叉验证估计，不会像单次 holdout 那样被反复调参抬高。

用法：
  python scripts/merge_cv.py --cvdir data/processed/cls_full_5fold --folds 5 \
      --out data/processed/cls_full_5fold/oof_pred.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(y, p, texts, th=0.5):
    from sklearn.metrics import f1_score, roc_auc_score
    y = np.asarray(y)
    preds = (np.asarray(p) >= th).astype(int)
    overall = f1_score(y, preds, average="macro", zero_division=0)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])
    if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
        short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
        combined = 0.7 * overall + 0.3 * short
    else:
        short, combined = float("nan"), overall
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"combined": float(combined), "macro_f1": float(overall),
            "short_f1": float(short), "auc": float(auc)}


def main():
    ap = argparse.ArgumentParser(description="合并 K 折 OOF 预测")
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(args.cvdir, "oof_pred.jsonl")

    rows = []
    for i in range(args.folds):
        p = os.path.join(args.cvdir, "pred_fold%d.jsonl" % i)
        if not os.path.exists(p):
            raise SystemExit("缺少第 %d 折预测: %s" % (i, p))
        part = load(p)
        print("fold %d: %d 条" % (i, len(part)))
        rows.extend(part)

    rows.sort(key=lambda r: int(r["id"]))
    ids = [int(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("OOF 存在重复 id，检查分折是否重叠")

    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["prob"]) for r in rows])
    texts = [r.get("text", "") for r in rows]

    m05 = score(y, p, texts, 0.5)
    best_th, best_comb = 0.5, -1.0
    for th in np.linspace(0.01, 0.99, 197):
        s = score(y, p, texts, float(th))["combined"]
        if s > best_comb:
            best_th, best_comb = float(th), float(s)

    print("\n=== OOF 汇总（%d 条，%d 折）===" % (len(rows), args.folds))
    print("AUC %.4f | 宏F1@0.5 %.4f | 短文本 %.4f | 综合@0.5 %.4f"
          % (m05["auc"], m05["macro_f1"], m05["short_f1"], m05["combined"]))
    print("最优阈值 %.3f -> 综合 %.4f" % (best_th, best_comb))

    fold_metrics = {}
    for i in range(args.folds):
        mp = os.path.join(args.cvdir, "metrics_fold%d.json" % i)
        if os.path.exists(mp):
            fold_metrics["fold%d" % i] = json.load(open(mp, encoding="utf-8"))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "n": len(rows), "folds": args.folds,
        "at_0.5": m05, "best_threshold": best_th, "best_combined": best_comb,
        "per_fold": fold_metrics, "out": os.path.abspath(out),
    }
    sp = os.path.join(args.cvdir, "cv_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("OOF 已写出:", out)
    print("汇总已写出:", sp)


if __name__ == "__main__":
    main()
