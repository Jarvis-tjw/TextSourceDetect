# -*- coding: utf-8 -*-
"""
数据统计脚本（analyze_data.py）
==============================
用途：把 data.jsonl 从头到尾扫一遍，统计真实分布，回答这些问题：
  - 总量多少？两套 schema 各占多少？有多少条带灰盒字段 probas/logits？
  - 领域(source)分布、生成器(model)分布、语言分布各是怎样？
  - human / machine 两类样本数量（看类别是否平衡）；
  - 文本长度分布，以及"短文本(<100 词)"占比（对应评分的 30% 子集）。

为什么要先统计、再建模？
  拿真实分布去决定：用不用多语模型、要不要均衡采样、短文本是否需特殊处理。

运行：
  python scripts/analyze_data.py
"""

import json
import os
import sys
from collections import Counter

# 让脚本无论从哪个目录运行都能 import 到同目录的 common.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else C.DATA_PATH
    print("正在扫描:", path)

    schema_cnt = Counter()   # 两套 schema 的计数
    src_cnt = Counter()      # 领域分布
    model_cnt = Counter()    # 生成器分布
    lang_cnt = Counter()     # 语言分布
    n_records = 0            # 记录总数
    n_graybox = 0            # 含 probas 或 logits 的记录数
    n_human = 0              # human 样本数（去重后）
    n_machine = 0            # machine 样本数
    human_words = []         # 每条 human 的词数
    machine_words = []       # 每条 machine 的词数
    seen_human = set()       # 用于按 (domain, source_id) 给 human 去重

    # jsonl 是一行一条 JSON；用流式读取，避免一次性吃满内存。
    # enumerate 提供行号，用于在源文章 id 缺失时做兜底唯一键，避免误判为同一篇。
    with open(path, encoding="utf-8") as f:
        for row_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_records += 1

            # 统计 schema 与灰盒字段覆盖情况
            schema_cnt[C.schema_of(rec)] += 1
            if "probas" in rec or "logits" in rec:
                n_graybox += 1

            domain = C.get_domain(rec)
            src_cnt[domain] += 1

            human = C.get_human_text(rec)
            machine = C.get_machine_text(rec)
            sid = C.get_source_id(rec) or ("__row%d" % row_no)  # id 缺失时用行号兜底

            # human 去重：同一篇源文章只算一次人类样本
            key = (domain, sid)
            if human.strip() and key not in seen_human:
                seen_human.add(key)
                n_human += 1
                human_words.append(C.count_words(human))
                lang_cnt[C.detect_lang(human)] += 1

            if machine.strip():
                n_machine += 1
                machine_words.append(C.count_words(machine))
                lang_cnt[C.detect_lang(machine)] += 1
                model_cnt[C.get_generator(rec)] += 1

    # -------------------- 打印统计报告 --------------------
    def top(counter, k=20):
        return counter.most_common(k)

    print("\n" + "=" * 60)
    print("总记录数:", n_records)
    print("schema 分布:", dict(schema_cnt))
    print("含灰盒字段(probas/logits)记录数:", n_graybox)
    print("human 样本数(去重后):", n_human, "| machine 样本数:", n_machine)
    if n_human:
        print("类别比例 machine:human = %.2f : 1" % (n_machine / n_human))
    print("\n[领域 source Top20]")
    for k, v in top(src_cnt):
        print("  %8d  %s" % (v, k))
    print("\n[生成器 model Top20]")
    for k, v in top(model_cnt):
        print("  %8d  %s" % (v, k))
    print("\n[语言分布]")
    for k, v in lang_cnt.most_common():
        print("  %8d  %s" % (v, k))

    def length_report(words, name):
        """打印某一类样本的词长分布，并给出"短文本(<100 词)"占比。"""
        if not words:
            return
        s = sorted(words)
        short = sum(1 for x in s if x < 100)
        print("\n[%s 词长] 条数=%d 均值=%.1f 中位=%d p10=%d p90=%d | 短文本(<100词)占比=%.1f%%"
              % (name, len(s), sum(s) / len(s), s[len(s) // 2],
                 s[len(s) // 10], s[len(s) * 9 // 10], 100.0 * short / len(s)))

    length_report(human_words, "human")
    length_report(machine_words, "machine")
    print("=" * 60)

    # -------------------- 落盘一份 JSON，方便后续脚本复用 --------------------
    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    summary = {
        "n_records": n_records,
        "schema": dict(schema_cnt),
        "graybox_records": n_graybox,
        "n_human": n_human,
        "n_machine": n_machine,
        "source": dict(src_cnt),
        "model": dict(model_cnt),
        "lang": dict(lang_cnt),
    }
    out = os.path.join(C.PROCESSED_DIR, "stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("统计结果已写入:", out)


if __name__ == "__main__":
    main()
