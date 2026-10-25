# -*- coding: utf-8 -*-
"""
Phase 1 统计基线（baseline_tfidf.py）
====================================
作用：用"最传统但很稳"的方法先跑通整个闭环，得到一份可提交的预测结果。
方法：TF-IDF 特征 + 逻辑回归（LogisticRegression）。

为什么先做它？
  1) 不依赖 GPU，几分钟就能跑完，用来验证"数据 -> 训练 -> 预测 -> 评测 -> 提交"全流程；
  2) 作为后续神经模型的性能下界对照；
  3) 它本身也可作为集成融合里的一个成员。

TF-IDF 用了两类特征拼起来：
  - 词级（word 1-2 gram）：抓关键词/用词；
  - 字符级（char_wb 3-5 gram）：抓标点、大小写、空格等"书写风格"，
    这类风格往往能区分"人写"与"机写"。

运行：
  python scripts/baseline_tfidf.py --max-train 40000
  说明：--max-train 限制训练样本量以加速首跑；0 表示用全量。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import eval as E


def parse_args():
    ap = argparse.ArgumentParser(description="TF-IDF + 逻辑回归 基线")
    ap.add_argument("--train", default=os.path.join(C.PROCESSED_DIR, "train.jsonl"))
    ap.add_argument("--holdout", default=os.path.join(C.PROCESSED_DIR, "holdout.jsonl"))
    ap.add_argument("--outdir", default=C.PROCESSED_DIR)
    ap.add_argument("--max-train", type=int, default=40000, help="最多用多少条训练样本，0=全量")
    ap.add_argument("--max-features", type=int, default=200000, help="TF-IDF 最大特征数")
    ap.add_argument("--save-model", default="", help="可选：把向量化器+分类器保存为 joblib，供融合推理复用")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_jsonl(path, max_n=0):
    """读取预处理后的 jsonl（每行 {text,label,...}），返回 texts 与 labels。"""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(int(r["label"]))
            if max_n and len(texts) >= max_n:
                break
    return texts, labels


def main():
    args = parse_args()
    t0 = time.time()

    print("读取训练集:", args.train)
    train_texts, train_labels = load_jsonl(args.train, args.max_train)
    print("  训练样本:", len(train_texts))

    print("读取 holdout:", args.holdout)
    hold_texts, hold_labels = load_jsonl(args.holdout, 0)
    print("  holdout 样本:", len(hold_texts))

    # ---------------- 特征：词级 + 字符级 TF-IDF 拼接 ----------------
    print("构建 TF-IDF 特征 ...")
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=args.max_features, min_df=2, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=args.max_features, min_df=2, sublinear_tf=True,
    )
    union = FeatureUnion([("word", word_vec), ("char", char_vec)])

    X_train = union.fit_transform(train_texts)
    X_hold = union.transform(hold_texts)
    print("  特征维度:", X_train.shape)

    # ---------------- 分类器：逻辑回归（对类别不平衡做加权） ----------------
    print("训练逻辑回归 ...")
    clf = LogisticRegression(
        max_iter=1000, C=4.0, class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_train, train_labels)

    # 保存模型：融合推理时需要在新数据上重新算 TF-IDF 概率，
    # 只存预测结果是不够的。
    if args.save_model:
        save_path = os.path.abspath(args.save_model)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"union": union, "clf": clf}, save_path)
        print("模型已保存:", save_path)

    # ---------------- 在 holdout 上预测并评估 ----------------
    probs = clf.predict_proba(X_hold)[:, 1]  # 取"是 machine(1)"的概率

    th, metrics = E.search_threshold(hold_labels, probs, hold_texts)
    print("holdout 最优阈值 %.2f -> %s" % (th, json.dumps(metrics, ensure_ascii=False)))

    # ---------------- 保存预测（供融合与提交复用） ----------------
    os.makedirs(args.outdir, exist_ok=True)
    pred_path = os.path.join(args.outdir, "baseline_pred_holdout.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (txt, y, p) in enumerate(zip(hold_texts, hold_labels, probs)):
            f.write(json.dumps({"id": i, "text": txt, "label": y, "prob": float(p)},
                               ensure_ascii=False) + "\n")
    print("预测已写出:", pred_path)
    print("耗时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()
