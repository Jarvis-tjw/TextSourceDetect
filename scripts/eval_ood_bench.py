# -*- coding: utf-8 -*-
"""
多域 OOD 评估台（eval_ood_bench.py）
====================================
【为什么需要它】
只用 arxiv 一个域做留出，结论很容易被单点噪声带偏：某次改动在 arxiv 上
涨 0.5 分，可能只是这一个域的偶然，换到 RuATD 或阿拉伯语维基就归零。
本脚本把多个"整体留出域"的结果汇总到同一张表，用来判断改动是否真的
跨域有效，而不是只对一个域有效。

【协议】
每个留出模型都在自己的"域内验证集"（indomain_val）上拟合 Platt 校准，
再在完全没见过的目标域上评估——校准和评估用的数据严格分离：
  - 校准：indomain_val（目标域之外）
  - 评估：eval_<domain>.jsonl（目标域，训练时完全排除）
这样目标域的标签不参与任何拟合，不会偷看答案。

主判据用 AUC（阈值无关，受校准偏移影响最小），辅以 oracle 综合分
（在目标域上"作弊"调阈值能达到的上限）和 0.5 阈值下的错误结构。

用法：
  python scripts/eval_ood_bench.py \
      --run arxiv=data/processed/bench_base/arxiv \
      --run RuATD_2022=data/processed/bench_base/RuATD_2022 \
      --run arabic-wikipedia=data/processed/bench_base/arabic-wikipedia
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from eval_ood import fit_platt, apply_platt, score, oracle_threshold


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(run_dir, domain):
    """在给定留出模型上算目标域指标。返回 dict 或 None（缺文件）。"""
    fit_path = os.path.join(run_dir, "pred_holdout.jsonl")
    test_path = os.path.join(run_dir, "pred_extra_eval_%s.jsonl" % domain)
    if not (os.path.exists(fit_path) and os.path.exists(test_path)):
        return None

    fit = load(fit_path)
    y_fit = np.array([int(r["label"]) for r in fit])
    p_fit = np.array([float(r["prob"]) for r in fit])
    a, b = fit_platt(y_fit, p_fit)

    test = load(test_path)
    y = np.array([int(r["label"]) for r in test])
    p = apply_platt(np.array([float(r["prob"]) for r in test]), a, b)
    texts = [r.get("text", "") for r in test]

    m = score(y, p, texts)
    th, best = oracle_threshold(y, p, texts)
    preds = p >= 0.5
    rec_m = float(preds[y == 1].mean()) if (y == 1).any() else float("nan")
    rec_h = float((~preds[y == 0]).mean()) if (y == 0).any() else float("nan")
    prec_m = (float((y[preds] == 1).mean()) if preds.any() else float("nan"))

    return {"domain": domain, "n": m["n"], "auc": m["auc"],
            "combined@0.5": m["combined"], "macro_f1@0.5": m["macro_f1"],
            "short_f1@0.5": m["short_f1"], "oracle_combined": best,
            "oracle_threshold": th, "recall_machine": rec_m,
            "recall_human": rec_h, "precision_machine": prec_m,
            "calib": [a, b], "run_dir": run_dir}


def main():
    ap = argparse.ArgumentParser(description="多域 OOD 汇总评估")
    ap.add_argument("--run", action="append", default=[],
                    help="DOMAIN=模型产物目录，可重复；目录需含 pred_holdout.jsonl 与 pred_extra_eval_<domain>.jsonl")
    ap.add_argument("--out", default="", help="结果 JSON 路径，留空则只打印")
    args = ap.parse_args()

    results = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit("--run 格式应为 DOMAIN=DIR，收到: %s" % spec)
        domain, run_dir = spec.split("=", 1)
        r = evaluate(run_dir.strip(), domain.strip())
        if r is None:
            print("跳过 %s：%s 缺少所需预测文件" % (domain, run_dir))
            continue
        results.append(r)

    if not results:
        raise SystemExit("没有任何可用结果，检查 --run 路径")

    print("\n多域 OOD 汇总（校准：域内验证集；评估：完全留出域）")
    print("=" * 96)
    print("%-22s %6s %8s %10s %10s %10s %9s %9s"
          % ("domain", "n", "AUC", "综合@0.5", "oracle综合", "macro@0.5", "M召回", "H召回"))
    for r in results:
        print("%-22s %6d %8.4f %10.4f %10.4f %10.4f %9.4f %9.4f"
              % (r["domain"], r["n"], r["auc"], r["combined@0.5"],
                 r["oracle_combined"], r["macro_f1@0.5"],
                 r["recall_machine"], r["recall_human"]))
    print("-" * 96)
    print("平均 AUC %.4f | 平均 oracle 综合 %.4f"
          % (np.mean([r["auc"] for r in results]),
             np.mean([r["oracle_combined"] for r in results])))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"runs": results,
                       "mean_auc": float(np.mean([r["auc"] for r in results])),
                       "mean_oracle_combined": float(np.mean([r["oracle_combined"] for r in results]))},
                      f, ensure_ascii=False, indent=2)
        print("已写出:", args.out)


if __name__ == "__main__":
    main()
