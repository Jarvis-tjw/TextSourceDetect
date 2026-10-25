# -*- coding: utf-8 -*-
"""
概率校准（calibrate.py）
========================
【要解决的问题】
Phase 2 训练时用了"逆频率类别权重"（给 machine 那一类额外加权），加上只训 2 个
epoch，模型输出的概率整体向 machine 偏移。后果是：

    本地最优阈值 = 0.97（不是 0.5）
    用 0.5 切分时，holdout 综合分只有 0.9406
    用 0.97 切分时，综合分是 0.9711

也就是说，如果官方评测按 0.5 把概率转成 0/1，我们会白丢约 3 分。

【校准为什么安全】
温度缩放（Temperature Scaling）和 Platt 缩放都是【单调变换】：
    z = log(p / (1-p))            # 把概率还原成 logit
    q = sigmoid(a*z + b)          # 单调递增
单调变换不改变任何一对样本的先后顺序，所以 AUC / 排序能力完全不变，
只是把概率的"刻度"和对零点重新对齐。它救不了排序错误，但能把 0.5 拉回最优边界。

【做法】
 1) 把 holdout 按标签分层切成两半：一半"拟合校准参数"，另一半"验证效果"。
    这样报告出来的提升不是自欺欺人。
 2) 同时拟合两种单调校准：温度缩放(a=1/T, b=0) 与 Platt(a, b)，
    用验证半边的对数损失(NLL)选更优的那个。
 3) 确认校准后"最优阈值"是否回到 0.5 附近。
 4) 用全部 holdout 重新拟合最终参数，写入 calibration.json，供 predict.py 使用。

用法：
  python scripts/calibrate.py
  python scripts/calibrate.py --pred data/processed/cls_xlmr_large_lora/pred_holdout.jsonl
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

EPS = 1e-6  # 防止 log(0)


def prob_to_logit(p):
    """概率 -> logit。裁掉极端值，避免 0/1 导致无穷大。"""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(z, y):
    """温度缩放：q = sigmoid(z / T)。只有一个参数 T>0。
    用 log 参数化保证 T 恒正，再用 L-BFGS 最小化对数损失。
    """
    def nll(log_t):
        t = np.exp(log_t[0])
        q = np.clip(sigmoid(z / t), EPS, 1 - EPS)
        return log_loss(y, q)

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B",
                   bounds=[(-3.0, 3.0)])
    t = float(np.exp(res.x[0]))
    return {"method": "temperature", "a": 1.0 / t, "b": 0.0, "T": t}


def fit_platt(z, y):
    """Platt 缩放：q = sigmoid(a*z + b)。
    比温度缩放多一个偏置 b，可以额外修正"整体偏高/偏低"，但也是单调的。
    """
    def nll(params):
        a, b = params
        q = np.clip(sigmoid(a * z + b), EPS, 1 - EPS)
        return log_loss(y, q)

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B",
                   bounds=[(0.01, 50.0), (-20.0, 20.0)])
    a, b = float(res.x[0]), float(res.x[1])
    return {"method": "platt", "a": a, "b": b, "T": 1.0 / a}


def apply_calibration(p, params):
    """按已拟合参数把原始概率映射成校准后概率。"""
    z = prob_to_logit(p)
    return sigmoid(params["a"] * z + params["b"])


def align_prior(p, params, target_prior):
    """把判定边界平移到目标先验上（只改截距 b，不改变排序）。

    为什么需要它：
      Platt 校准只保证"概率刻度正确"，但当验证集/官方 val 的正例比例
      与拟合数据不同时，0.5 这个切点会整体偏掉。
      实测 OOD 场景下模型预测正例率只有 40.6%，而真实先验是 58.1%，
      结果大量机器样本被漏判。

    做法：
      1) 在校准后概率上取 (1 - target_prior) 分位点 t*；
      2) 给截距加上 -logit(t*)，使 t* 恰好映射到 0.5。
      这样官方按 0.5 切分时，预测正例率就等于目标先验。
      纯截距平移仍是单调变换，AUC 完全不变。
    """
    q = apply_calibration(p, params)
    t_star = float(np.quantile(q, 1.0 - target_prior))
    shift = float(-prob_to_logit(np.array([t_star]))[0])
    out = dict(params)
    out["b"] = float(params["b"] + shift)
    out["prior_target"] = float(target_prior)
    out["prior_shift"] = shift
    out["prior_threshold"] = t_star
    return out


def search_threshold(y, p, texts, lo=0.01, hi=0.995, steps=198):
    """网格搜索使赛题综合分最高的阈值。
    这里同样处理"短文本子集只有单一类别"的伪影：此时不计入 30% 权重。
    """
    y = np.asarray(y)
    p = np.asarray(p)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])

    def score(th):
        preds = (p >= th).astype(int)
        overall = f1_score(y, preds, average="macro", zero_division=0)
        if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
            short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
            return 0.7 * overall + 0.3 * short, overall, short
        return overall, overall, float("nan")

    best_th, best = 0.5, (-1.0, 0.0, 0.0)
    for th in np.linspace(lo, hi, steps):
        s = score(float(th))
        if s[0] > best[0]:
            best_th, best = float(th), s
    return best_th, {"combined": float(best[0]), "macro_f1": float(best[1]),
                     "short_macro_f1": float(best[2])}


def metrics_at(y, p, texts, th):
    preds = (np.asarray(p) >= th).astype(int)
    y = np.asarray(y)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])
    overall = f1_score(y, preds, average="macro", zero_division=0)
    if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
        short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
        combined = 0.7 * overall + 0.3 * short
    else:
        short, combined = float("nan"), overall
    return {"combined": float(combined), "macro_f1": float(overall),
            "short_macro_f1": float(short), "threshold": float(th)}


def ece(y, p, n_bins=10):
    """期望校准误差：把概率分箱，比较"预测均值"与"实际正例率"。
    这个数会写进技术报告，用来量化校准前后的改善（越小越好）。
    """
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser(description="概率校准（温度缩放 / Platt）")
    ap.add_argument("--pred", default="data/processed/cls_xlmr_large_lora/pred_holdout.jsonl")
    ap.add_argument("--out", default="data/processed/cls_xlmr_large_lora/calibration.json")
    ap.add_argument("--target-prior", type=float, default=0.0,
                    help="目标正例先验；0 表示用拟合数据的真实比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["prob"]) for r in rows])
    texts = [r.get("text", "") for r in rows]
    print("样本数:", len(y), "| 原始 AUC: %.4f" % roc_auc_score(y, p))

    # ---------- 1. 分层切分：一半拟合、一半验证 ----------
    rng = np.random.RandomState(args.seed)
    idx_fit, idx_val = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = len(idx) // 2
        idx_fit.extend(idx[:cut])
        idx_val.extend(idx[cut:])
    idx_fit, idx_val = np.array(idx_fit), np.array(idx_val)

    z = prob_to_logit(p)
    print("拟合半边 %d 条 | 验证半边 %d 条" % (len(idx_fit), len(idx_val)))

    # ---------- 2. 两种单调校准，按验证半边 NLL 选优 ----------
    candidates = {}
    for name, fn in (("temperature", fit_temperature), ("platt", fit_platt)):
        params = fn(z[idx_fit], y[idx_fit])
        q_val = apply_calibration(p[idx_val], params)
        candidates[name] = {
            "params": params,
            "val_nll": float(log_loss(y[idx_val], np.clip(q_val, EPS, 1 - EPS))),
        }
        print("  %-11s a=%.4f b=%.4f T=%.4f | 验证半边 NLL=%.5f"
              % (name, params["a"], params["b"], params["T"], candidates[name]["val_nll"]))

    best_name = min(candidates, key=lambda k: candidates[k]["val_nll"])
    best = candidates[best_name]["params"]
    print("选择:", best_name)

    # 目标先验只用"拟合半边"的标签估计，避免偷看验证半边
    prior_fit = float(np.mean(y[idx_fit])) if args.target_prior <= 0 else float(args.target_prior)

    # ---------- 3. 在验证半边对比校准前后 ----------
    p_before = p[idx_val]
    p_after = apply_calibration(p_before, best)
    best_prior = align_prior(p[idx_fit], best, prior_fit)
    p_after_prior = apply_calibration(p_before, best_prior)
    y_val = y[idx_val]
    t_val = [texts[i] for i in idx_val]

    th_before, m_before = search_threshold(y_val, p_before, t_val)
    th_after, m_after = search_threshold(y_val, p_after, t_val)
    at05_before = metrics_at(y_val, p_before, t_val, 0.5)
    at05_after = metrics_at(y_val, p_after, t_val, 0.5)
    at05_prior = metrics_at(y_val, p_after_prior, t_val, 0.5)

    print("\n--- 验证半边（独立的 50%，未参与拟合） ---")
    print("校准前: 最优阈值 %.3f | 该阈值综合分 %.4f | 用0.5切分综合分 %.4f"
          % (th_before, m_before["combined"], at05_before["combined"]))
    print("校准后: 最优阈值 %.3f | 该阈值综合分 %.4f | 用0.5切分综合分 %.4f"
          % (th_after, m_after["combined"], at05_after["combined"]))
    print("先验对齐后: 0.5切分综合分 %.4f | 目标先验 %.4f | 预测正例率 %.4f"
          % (at05_prior["combined"], prior_fit, float((p_after_prior >= 0.5).mean())))
    print("最优分变化 %+0.4f | 0.5切分变化 %+0.4f | 先验对齐相对校准 %+0.4f"
          % (m_after["combined"] - m_before["combined"],
             at05_after["combined"] - at05_before["combined"],
             at05_prior["combined"] - at05_after["combined"]))
    print("ECE: 校准前 %.4f -> 校准后 %.4f -> 先验对齐后 %.4f"
          % (ece(y_val, p_before), ece(y_val, p_after), ece(y_val, p_after_prior)))
    print("AUC: 校准前 %.4f -> 校准后 %.4f（单调变换，理论上应完全不变）"
          % (roc_auc_score(y_val, p_before), roc_auc_score(y_val, p_after)))

    # ---------- 4. 用全部 holdout 重新拟合最终参数 ----------
    final = fit_temperature(z, y) if best_name == "temperature" else fit_platt(z, y)
    final_prior = float(np.mean(y)) if args.target_prior <= 0 else float(args.target_prior)
    final_aligned = align_prior(p, final, final_prior)
    p_all = apply_calibration(p, final_aligned)
    th_all, m_all = search_threshold(y, p_all, texts)
    at05_all = metrics_at(y, p_all, texts, 0.5)

    print("\n--- 全量 holdout 最终参数（先验对齐后） ---")
    print("a=%.6f b=%.6f T=%.6f | 目标先验 %.4f | 截距平移 %+0.4f"
          % (final_aligned["a"], final_aligned["b"], final_aligned["T"],
             final_prior, final_aligned["prior_shift"]))
    print("未对齐参数 a=%.6f b=%.6f" % (final["a"], final["b"]))
    print("全量最优阈值 %.3f | 该阈值综合分 %.4f | 0.5切分综合分 %.4f | 预测正例率 %.4f"
          % (th_all, m_all["combined"], at05_all["combined"],
             float((p_all >= 0.5).mean())))
    print("全量 ECE: 原始 %.4f -> 校准后 %.4f -> 先验对齐后 %.4f"
          % (ece(y, p), ece(y, apply_calibration(p, final)), ece(y, p_all)))

    out = {
        "method": final_aligned["method"],
        "a": final_aligned["a"],
        "b": final_aligned["b"],
        "T": final_aligned["T"],
        "a_raw": final["a"],
        "b_raw": final["b"],
        "prior_target": final_aligned["prior_target"],
        "prior_shift": final_aligned["prior_shift"],
        "prior_threshold": final_aligned["prior_threshold"],
        "fit_on": os.path.abspath(args.pred),
        "n_fit": int(len(y)),
        "validation": {
            "split_seed": args.seed,
            "n_val": int(len(idx_val)),
            "best_method": best_name,
            "val_nll": candidates[best_name]["val_nll"],
            "threshold_before": th_before,
            "threshold_after": th_after,
            "combined_before_own_th": m_before["combined"],
            "combined_after_own_th": m_after["combined"],
            "combined_before_at_0.5": at05_before["combined"],
            "combined_after_at_0.5": at05_after["combined"],
            "combined_prior_at_0.5": at05_prior["combined"],
            "prior_target_fit_half": prior_fit,
            "ece_before": ece(y_val, p_before),
            "ece_after": ece(y_val, p_after),
            "ece_prior": ece(y_val, p_after_prior),
        },
        "full_holdout": {
            "threshold_after_calibration": th_all,
            "combined_after_calibration": m_all["combined"],
            "combined_at_0.5": at05_all["combined"],
            "positive_rate_at_0.5": float((p_all >= 0.5).mean()),
            "ece_before": ece(y, p),
            "ece_after": ece(y, p_all),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n校准参数已写出:", args.out)
    print("提示：predict.py 会自动读取该文件并应用校准。")


if __name__ == "__main__":
    main()
