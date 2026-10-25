# -*- coding: utf-8 -*-
"""
真 OOD 评估（eval_ood.py）
==========================
【这个脚本回答什么问题】
"如果验证集是一个训练时完全没见过的领域，模型会掉多少分？"

对比两个模型，在【同一个领域 arxiv】上的表现：
  A. 基线：见过 arxiv 的同分布模型（Phase 2 主力）
  B. OOD ：训练时把 arxiv 整体挖掉的模型

为了让比较公平，两者都采用同一套"只看不到目标域标签"的流程：
  1) 在【目标域之外】的数据上拟合 Platt 概率校准；
  2) 直接用 0.5 切分目标域，算赛题三项指标。
（不能各自在目标域上调阈值，那样等于偷看答案。）

【输入】
  --baseline-pred      基线模型在 holdout 上的预测（含 label/text/prob）
  --baseline-meta      与上面逐行对应的 holdout 元信息（含 domain）
  --ood-train-pred     OOD 模型在其"域内验证集"上的预测（用于拟合校准）
  --ood-test-pred      OOD 模型在 arxiv 上的预测
  --domain             目标领域名，默认 arxiv

用法：
  python scripts/eval_ood.py
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

EPS = 1e-6


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(y, p):
    """在给定数据上拟合 Platt 缩放，返回 (a, b)。"""
    z = logit(p)
    y = np.asarray(y)

    def nll(params):
        a, b = params
        q = np.clip(sigmoid(a * z + b), EPS, 1 - EPS)
        return log_loss(y, q)

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B",
                   bounds=[(0.01, 50.0), (-20.0, 20.0)])
    return float(res.x[0]), float(res.x[1])


def apply_platt(p, a, b):
    return sigmoid(a * logit(p) + b)


def score(y, p, texts, th=0.5):
    """赛题口径：宏F1 / 短文本F1 / 0.7+0.3 综合分，并修正单类短文本子集。"""
    y = np.asarray(y)
    preds = (np.asarray(p) >= th).astype(int)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])

    overall = f1_score(y, preds, average="macro", zero_division=0)
    if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
        short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
        combined = 0.7 * overall + 0.3 * short
    else:
        short, combined = float("nan"), overall
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"n": len(y), "macro_f1": overall, "short_f1": short,
            "combined": combined, "auc": auc}


def oracle_threshold(y, p, texts):
    """如果在目标域上"作弊"调阈值，最好能到多少？用于衡量校准迁移损失。"""
    best_th, best = 0.5, -1.0
    for th in np.linspace(0.01, 0.995, 198):
        s = score(y, p, texts, float(th))["combined"]
        if s > best:
            best_th, best = float(th), s
    return best_th, best


def main():
    ap = argparse.ArgumentParser(description="真 OOD 对比评估")
    ap.add_argument("--baseline-pred", default="data/processed/cls_xlmr_large_lora/pred_holdout.jsonl")
    ap.add_argument("--baseline-meta", default="data/processed/holdout.jsonl")
    ap.add_argument("--ood-train-pred", default="data/processed/cls_ood_no_arxiv/pred_holdout.jsonl")
    ap.add_argument("--ood-test-pred", default="data/processed/cls_ood_no_arxiv/pred_arxiv_holdout.jsonl")
    ap.add_argument("--domain", default="arxiv")
    ap.add_argument("--out", default="data/processed/ood/ood_report.json")
    args = ap.parse_args()

    for p in (args.baseline_pred, args.baseline_meta, args.ood_train_pred, args.ood_test_pred):
        if not os.path.exists(p):
            raise SystemExit("缺少输入文件: %s" % p)

    # ---------- 基线模型：在非目标域上拟合校准，在目标域上评估 ----------
    base = load(args.baseline_pred)
    meta = load(args.baseline_meta)
    if len(base) != len(meta):
        raise SystemExit("基线与元信息行数不一致: %d vs %d" % (len(base), len(meta)))

    dom = np.array([m.get("domain", "unknown") for m in meta])
    is_target = dom == args.domain

    y_all = np.array([int(r["label"]) for r in base])
    p_all = np.array([float(r["prob"]) for r in base])
    t_all = np.array([r.get("text", "") for r in base])

    a, b = fit_platt(y_all[~is_target], p_all[~is_target])
    p_base = apply_platt(p_all[is_target], a, b)
    m_base = score(y_all[is_target], p_base, t_all[is_target])
    th_base, best_base = oracle_threshold(y_all[is_target], p_base, t_all[is_target])
    print("基线模型（见过 %s）: 校准 a=%.4f b=%.4f" % (args.domain, a, b))
    print("  目标域 n=%d | AUC %.4f | 宏F1 %.4f | 短文本F1 %.4f | 综合 %.4f"
          % (m_base["n"], m_base["auc"], m_base["macro_f1"], m_base["short_f1"], m_base["combined"]))
    print("  （若允许在目标域上调阈值: th=%.3f -> %.4f）" % (th_base, best_base))

    # ---------- OOD 模型：在其域内验证集上拟合校准，在目标域上评估 ----------
    ood_fit = load(args.ood_train_pred)
    y_fit = np.array([int(r["label"]) for r in ood_fit])
    p_fit = np.array([float(r["prob"]) for r in ood_fit])
    a2, b2 = fit_platt(y_fit, p_fit)

    ood_test = load(args.ood_test_pred)
    y_ood = np.array([int(r["label"]) for r in ood_test])
    p_ood = apply_platt(np.array([float(r["prob"]) for r in ood_test]), a2, b2)
    t_ood = np.array([r.get("text", "") for r in ood_test])
    m_ood = score(y_ood, p_ood, t_ood)
    th_ood, best_ood = oracle_threshold(y_ood, p_ood, t_ood)
    print("\nOOD 模型（训练时完全排除 %s）: 校准 a=%.4f b=%.4f" % (args.domain, a2, b2))
    print("  目标域 n=%d | AUC %.4f | 宏F1 %.4f | 短文本F1 %.4f | 综合 %.4f"
          % (m_ood["n"], m_ood["auc"], m_ood["macro_f1"], m_ood["short_f1"], m_ood["combined"]))
    print("  （若允许在目标域上调阈值: th=%.3f -> %.4f）" % (th_ood, best_ood))

    # ---------- 结论 ----------
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    drop = m_base["combined"] - m_ood["combined"]
    print("同分布(见过该域) 综合 %.4f -> 真OOD(没见过) 综合 %.4f | 掉分 %.4f"
          % (m_base["combined"], m_ood["combined"], drop))
    print("AUC: %.4f -> %.4f | 掉 %.4f"
          % (m_base["auc"], m_ood["auc"], m_base["auc"] - m_ood["auc"]))
    if drop < 0.02:
        print("判读：掉分很小（<2 分），说明模型学到的是可迁移的信号，不是领域捷径。")
        print("      后续优化的边际收益低，可以把精力放在报告与交付上。")
    elif drop < 0.05:
        print("判读：掉分中等（2~5 分），跨域泛化有提升空间。")
        print("      建议做长度捷径削弱或双模型集成，并复测本脚本。")
    else:
        print("判读：掉分较大（>5 分），模型明显依赖领域内捷径。")
        print("      应优先做数据增强（随机截断/长度归一化）与 OOD 正则化。")
    print("校准迁移损失（0.5 -> 目标域最优阈值）: 基线 %.4f | OOD 模型 %.4f"
          % (best_base - m_base["combined"], best_ood - m_ood["combined"]))

    report = {
        "domain": args.domain,
        "baseline": {"metrics": m_base, "calib": [a, b], "oracle_threshold": th_base, "oracle_combined": best_base},
        "ood": {"metrics": m_ood, "calib": [a2, b2], "oracle_threshold": th_ood, "oracle_combined": best_ood},
        "drop_combined": drop,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n已写出:", args.out)


if __name__ == "__main__":
    main()
