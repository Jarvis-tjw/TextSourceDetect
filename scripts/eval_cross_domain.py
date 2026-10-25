# -*- coding: utf-8 -*-
"""
跨域 / 跨语言 / 跨长度 诊断脚本（eval_cross_domain.py）
=====================================================
【为什么需要这个脚本】
Phase 2 的 holdout 是从训练集里"随机按组切"出来的，所以它的领域比例、
生成器比例、语言比例和训练集几乎一样。在这种"同分布"数据上得到的
宏 F1 = 0.967，天然偏乐观。真实评测集通常是"没见过的领域/生成器/语言"，
分数会掉。

本脚本不重新训练、不重新推理，只做两件事：
  1) 把已有的 pred_holdout.jsonl 按 domain / lang / generator 切片，
     看模型究竟在哪些子集上变差；
  2) 做"阈值迁移实验"：在 A 领域上选出的阈值，直接拿到 B 领域用，
     看掉多少分 —— 这是对"真实验证集是另一个域"最便宜的模拟。

【为什么这比马上做 PPL/GLTR 更重要】
如果诊断显示：
  - 各领域 AUC 都很高、只是最优阈值不同  -> 问题在校准，用温度缩放就能救；
  - 某些领域 AUC 本身就很低              -> 问题在表征，才需要 PPL/GLTR/零样本去补。
先分清楚是哪一种，才不会白做 Phase 3。

【输入】
  --pred   预测文件（每行 {"id","text","label","prob"}），如
           data/processed/cls_xlmr_large_lora/pred_holdout.jsonl
  --meta   带元信息的原始 holdout（每行含 domain/lang/generator），如
           data/processed/holdout.jsonl
  两个文件必须"行序一一对应"（训练脚本是按顺序推理并写出的）。
  脚本会逐行比对 text 前缀，若对不上就报错，避免悄悄错位。

【输出】
  JSON 汇总 + 控制台表格。包含：
    - 全局最优阈值（网格扩到 0.995，顺便检查是否还卡在上界）
    - 每个 domain 的 n / AUC / 宏F1(全局阈值) / 该域自己最优阈值与分数
    - 每个 lang 的同样指标
    - 阈值迁移矩阵：用"除该域外全部数据"选的阈值，应用到该域
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import eval as E   # 复用赛题口径：宏F1 + 短文本F1 + 0.7/0.3 加权


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def align(pred_rows, meta_rows):
    """把预测结果和元信息按行序对齐，并校验文本一致。
    为什么要有校验：一旦两个文件顺序不同，后面的"按领域切片"就全错了，
    而且这种错误不会报错、只会给出看着很合理的错误结论。所以必须显式检查。
    """
    if len(pred_rows) != len(meta_rows):
        raise SystemExit("行数不一致：pred=%d meta=%d" % (len(pred_rows), len(meta_rows)))
    bad = 0
    for a, b in zip(pred_rows, meta_rows):
        # 只比前 40 个字符，足够判断是否错位，也不受截断差异影响
        if a.get("text", "")[:40] != b.get("text", "")[:40]:
            bad += 1
    if bad > 0:
        raise SystemExit("检测到 %d 行文本不匹配，两个文件行序可能不同，请勿继续。" % bad)


def _combined_score(y, preds, mask_short):
    """赛题综合分 = 0.7*整体宏F1 + 0.3*短文本宏F1，但做一个安全性修正：
    若"短文本子集"只有单一类别，其宏 F1 恒约 0.5，是纯数学伪影。
    此时不把这一项计入 30% 权重，直接退化为整体宏 F1，避免把伪影当成模型缺陷。
    """
    overall = f1_score(y, preds, average="macro", zero_division=0)
    if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
        short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
        return 0.7 * overall + 0.3 * short, overall, short
    return overall, overall, float("nan")


def best_threshold(y, p, texts, lo=0.01, hi=0.995, steps=198):
    """在给定子集上搜索使"赛题综合分"最高的阈值。
    注意：网格右端取 0.995（而不是之前的 0.95），因为 Phase 2 的最优阈值
    恰好落在 0.95 这个旧上界，有"被截断"的嫌疑，这里必须放开确认。
    """
    y = np.asarray(y)
    p = np.asarray(p)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])

    best_th, best_score = 0.5, -1.0
    for th in np.linspace(lo, hi, steps):
        preds = (p >= th).astype(int)
        score, _, _ = _combined_score(y, preds, mask_short)
        if score > best_score:
            best_score, best_th = score, float(th)
    return best_th, float(best_score)


def macro_f1_at(y, p, texts, th):
    """固定阈值下的赛题口径三项指标（含单类短文本子集的安全性修正）。"""
    y = np.asarray(y)
    preds = (np.asarray(p) >= th).astype(int)
    mask_short = np.array([C.count_words(t) < 100 for t in texts])
    combined, overall, short = _combined_score(y, preds, mask_short)
    return {
        "combined": float(combined),
        "macro_f1": float(overall),
        "short_macro_f1": float(short),
        "short_ratio": float(mask_short.mean()),
        "n": int(len(y)),
    }


def safe_auc(y, p):
    """AUC 衡量"排序能力"，与阈值无关。
    - AUC 高 + 固定阈值 F1 低  => 只是概率标定偏了（可校准）
    - AUC 本身低               => 模型区分不开（要换特征/换模型）
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def slice_report(name, keys, y, p, texts, global_th, min_group=30):
    """对某一种切分维度（domain/lang/generator）逐组统计。

    【重要口径修正】
    如果某个子集只有单一类别（例如某个生成器组全是 machine，human 组全是 human），
    在这个子集上算"宏平均 F1"是没有意义的：另一类的 TP=0，F1=0，
    宏平均会被硬拉到 0.5 左右，看起来像"模型全错"，其实只是指标不适用。
    因此这里分两种情况：
      - 两类都有：正常算宏 F1 / 短文本 F1 / 综合分 / AUC
      - 只有一类：改算"recall"（该类被正确判对的比例），并标 classes=1
    """
    keys = np.asarray(keys)
    y = np.asarray(y)
    p = np.asarray(p)
    texts = np.asarray(texts)

    rows = []
    for k in sorted(set(keys.tolist())):
        idx = np.where(keys == k)[0]
        if len(idx) < min_group:        # 太小的组统计不可靠，直接跳过
            continue
        yy, pp, tt = y[idx], p[idx], texts[idx]
        n_class = len(np.unique(yy))

        if n_class >= 2:
            fixed = macro_f1_at(yy, pp, tt, global_th)
            own_th, own_score = best_threshold(yy, pp, tt)
            rows.append({
                "group": str(k),
                "n": int(len(idx)),
                "classes": 2,
                "auc": safe_auc(yy, pp),
                "score@global": fixed["macro_f1"],
                "score_kind": "macro_f1",
                "short@global": fixed["short_macro_f1"],
                "own_threshold": own_th,
                "own_best": own_score,
                # gap 越大，说明这个域的最优判定边界越偏离全局阈值
                "gap_from_global": own_score - fixed["combined"],
            })
        else:
            # 单类子集：只报 recall（= 正确判对的比例），不报宏 F1、不报综合分
            label = int(yy[0])
            recall = float(((pp >= global_th).astype(int) == label).mean())
            rows.append({
                "group": str(k),
                "n": int(len(idx)),
                "classes": 1,
                "auc": float("nan"),
                "score@global": recall,
                "score_kind": "recall",
                "short@global": float("nan"),
                "own_threshold": float("nan"),
                "own_best": float("nan"),
                "gap_from_global": float("nan"),
            })
    return rows


def threshold_transfer(name, keys, y, p, texts):
    """阈值迁移实验：对每个组 g，
       1) 在"除 g 以外的所有数据"上选最优阈值 th_{-g}
       2) 把 th_{-g} 直接用到 g 上，算综合分
       3) 与"在 g 上自己选阈值"的最优分比较，差值 = 迁移损失
    这正是"真实评测集是一个新领域"时会发生的事的近似模拟。
    """
    keys = np.asarray(keys)
    y = np.asarray(y)
    p = np.asarray(p)
    texts = np.asarray(texts)

    rows = []
    for k in sorted(set(keys.tolist())):
        idx = np.where(keys == k)[0]
        if len(idx) < 30:
            continue
        # 单类子集不做阈值迁移：只有一类时，阈值高低只改变 recall，
        # "最优阈值"没有意义（永远会滑到极端），算出来的迁移损失是伪影。
        if len(np.unique(y[idx])) < 2:
            continue
        mask_out = keys != k
        if len(np.unique(y[mask_out])) < 2:
            continue

        th_out, _ = best_threshold(y[mask_out], p[mask_out], texts[mask_out])
        transferred = macro_f1_at(y[idx], p[idx], texts[idx], th_out)
        own_th, own_score = best_threshold(y[idx], p[idx], texts[idx])

        rows.append({
            "group": str(k),
            "n": int(len(idx)),
            "threshold_from_others": th_out,
            "combined_transferred": transferred["combined"],
            "macro_f1_transferred": transferred["macro_f1"],
            "combined_oracle": own_score,
            "transfer_loss": own_score - transferred["combined"],
        })
    return rows


def print_table(title, rows, cols):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if not rows:
        print("(无满足最小样本量的组)")
        return
    # 简单等宽排版
    widths = []
    for c in cols:
        w = max(len(c), max(len(_fmt(r.get(c))) for r in rows))
        widths.append(w)
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)))


def _fmt(v):
    if isinstance(v, float):
        return "%.4f" % v
    if v is None:
        return "-"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description="跨域/跨语言诊断（不重新训练）")
    ap.add_argument("--pred", default="data/processed/cls_xlmr_large_lora/pred_holdout.jsonl")
    ap.add_argument("--meta", default="data/processed/holdout.jsonl")
    ap.add_argument("--out", default="data/processed/cls_xlmr_large_lora/cross_domain_report.json")
    ap.add_argument("--min-group", type=int, default=30, help="小于该样本量的组不统计")
    args = ap.parse_args()

    print("读取预测:", args.pred)
    pred_rows = load_jsonl(args.pred)
    print("读取元信息:", args.meta)
    meta_rows = load_jsonl(args.meta)
    align(pred_rows, meta_rows)

    y = np.array([int(r["label"]) for r in pred_rows])
    p = np.array([float(r["prob"]) for r in pred_rows])
    texts = np.array([r.get("text", "") for r in pred_rows])
    domains = np.array([r.get("domain", "unknown") for r in meta_rows])
    langs = np.array([r.get("lang", "unknown") for r in meta_rows])
    gens = np.array([r.get("generator", "unknown") for r in meta_rows])

    print("对齐成功，样本数:", len(y))

    # ---------- 1. 全局阈值（放宽上界到 0.995，检查是否仍卡边界） ----------
    global_th, global_score = best_threshold(y, p, texts)
    gm = macro_f1_at(y, p, texts, global_th)
    print("\n全局最优阈值: %.4f (综合分 %.4f, 宏F1 %.4f, 短文本F1 %.4f)"
          % (global_th, global_score, gm["macro_f1"], gm["short_macro_f1"]))
    if global_th >= 0.99:
        print("  [警告] 最优阈值仍贴在搜索网格右端，概率校准偏向 machine，"
              "真实数据上需要重新校准（建议温度缩放）。")
    print("全局 AUC: %.4f" % safe_auc(y, p))

    # ---------- 2. 各维度切片 ----------
    dom_rows = slice_report("domain", domains, y, p, texts, global_th)
    lang_rows = slice_report("lang", langs, y, p, texts, global_th)
    gen_rows = slice_report("generator", gens, y, p, texts, global_th)

    cols_a = ["group", "n", "classes", "auc", "score@global", "score_kind", "short@global", "own_threshold", "own_best", "gap_from_global"]
    print_table("按领域（domain）", dom_rows, cols_a)
    print_table("按语言（lang）", lang_rows, cols_a)
    print_table("按生成器（generator）", gen_rows, cols_a)
    print("注：classes=1 的组只有单一类别，宏 F1 在数学上恒约 0.5，故改报 recall（该类判对比例）。")
    print("    生成器表的意义是：每个生成器被漏判多少、human 被误判多少。")

    # ---------- 3. 阈值迁移 ----------
    t_dom = threshold_transfer("domain", domains, y, p, texts)
    t_lang = threshold_transfer("lang", langs, y, p, texts)
    cols_b = ["group", "n", "threshold_from_others", "combined_transferred", "combined_oracle", "transfer_loss"]
    print_table("阈值迁移：用【其他域】的阈值判【本域】", t_dom, cols_b)
    print_table("阈值迁移：用【其他语言】的阈值判【本语言】", t_lang, cols_b)

    # ---------- 4. 汇总 ----------
    report = {
        "n": int(len(y)),
        "global": {
            "threshold": global_th,
            "combined": global_score,
            "macro_f1": gm["macro_f1"],
            "short_macro_f1": gm["short_macro_f1"],
            "auc": safe_auc(y, p),
        },
        "by_domain": dom_rows,
        "by_lang": lang_rows,
        "by_generator": gen_rows,
        "transfer_domain": t_dom,
        "transfer_lang": t_lang,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n已写出:", args.out)

    # ---------- 5. 给出结论性判断 ----------
    print("\n" + "=" * 78)
    print("判读建议")
    print("=" * 78)
    if dom_rows:
        two_class = [r for r in dom_rows if r["classes"] == 2]
        if two_class:
            worst = min(two_class, key=lambda r: r["auc"])
            best = max(two_class, key=lambda r: r["auc"])
            print("AUC 最高域: %s (%.4f) | 最低域: %s (%.4f) | 极差 %.4f"
                  % (best["group"], best["auc"], worst["group"], worst["auc"], best["auc"] - worst["auc"]))
    if gen_rows:
        # 生成器组都是单类，看谁的 recall 最低 —— 这就是"最容易漏判的生成器"
        worst_gen = min(gen_rows, key=lambda r: r["score@global"])
        print("最易漏判的生成器: %s (recall %.4f)" % (worst_gen["group"], worst_gen["score@global"]))
    if t_dom:
        avg_loss = float(np.mean([r["transfer_loss"] for r in t_dom]))
        worst_t = max(t_dom, key=lambda r: r["transfer_loss"])
        print("阈值迁移平均损失: %.4f | 最差域: %s (损失 %.4f)"
              % (avg_loss, worst_t["group"], worst_t["transfer_loss"]))
        print("若平均迁移损失 > 0.01 或最差域 > 0.03，说明阈值校准是主要风险，"
              "应优先做温度缩放，而不是先上 PPL/GLTR。")


if __name__ == "__main__":
    main()
