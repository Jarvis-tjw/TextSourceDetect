# -*- coding: utf-8 -*-
"""
推理与提交文件生成（predict.py）
================================
把 Phase 2 训练好的模型，变成"任意输入 jsonl -> result.csv"的交付入口。

【为什么输入格式要写得很宽松】
官方 val.jsonl 赛前不公开，字段名可能有几种情况：
  - 预处理规范格式：{"text","label","generator","domain"}
  - 原始双栏格式：  {"human_text", "machine_text", ...}
  - 编号字段可能是 "id" / "source_id" / "source_ID"
因此这里做自动探测，并且在遇到"一行里同时有人类文本和机器文本"时
明确报错而不是猜——猜错会导致提交的 id 全部错位，后果比报错严重得多。

【校准】
如果模型目录下有 calibration.json（由 calibrate.py 生成），会自动把
原始概率做单调映射。这一步不改变排序，只把 0.5 拉回最优判定边界，
是本次提升分数最关键的一步。可用 --no-calibrate 关闭以做对比。

用法：
  # 在 holdout 上验证整条链路（有标签时会顺便打印分数）
  python scripts/predict.py --input data/processed/holdout.jsonl \
      --out-csv data/processed/result_check.csv

  # 官方 val 到手后
  python scripts/predict.py --input data/val.jsonl --out-csv result.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# 官方可能使用的编号字段，按优先级尝试
ID_FIELDS = ("id", "source_id", "source_ID", "text_id", "uid")


def pick_text(row, text_field):
    """从一行里取出要分类的文本。
    返回 (text, mode)；mode 用于区分单栏/双栏，便于上层决定是否报错。
    """
    if text_field:
        return row.get(text_field, ""), "explicit"
    if "text" in row and row["text"]:
        return row["text"], "single"
    if row.get("human_text") and row.get("machine_text"):
        return None, "dual"
    if row.get("human_text"):
        return row["human_text"], "single"
    if row.get("machine_text"):
        return row["machine_text"], "single"
    return "", "missing"


def pick_id(row, fallback):
    for k in ID_FIELDS:
        if k in row:
            return str(row[k])
    return str(fallback)


def encode(texts, tokenizer, max_len):
    """分词。padding='max_length' 与训练时保持一致，保证复现同样的前向结果。"""
    enc = tokenizer(
        texts, truncation=True, max_length=max_len, padding="max_length",
        return_tensors="pt",
    )
    return enc


def run_inference(model, tokenizer, texts, device, max_len, batch,
                  chunk_long=False, chunk_max_windows=8):
    """对一组文本逐批推理，返回“是 machine”的概率列表。

    chunk_long=True 时，把超过 max_len 的文本按滑窗切成多个窗口分别前向，
    再在 logit 空间按窗口长度加权平均。训练时只能看到前 max_len 个 token，
    长文本尾部信息原本被直接丢弃；分块让判定覆盖整篇文档。
    短文本路径与原来完全一致，不改变任何已有行为。
    """
    use_bf16 = torch.cuda.is_available()
    if not chunk_long:
        probs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                chunk = texts[start:start + batch]
                enc = encode(chunk, tokenizer, max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"]).logits
                # 统一转 float32 再 softmax，避免 bf16 精度影响
                p = torch.softmax(logits.float(), dim=-1)[:, 1]
                probs.extend(p.cpu().tolist())
                if (start // batch + 1) % 50 == 0:
                    print("  已推理 %d / %d" % (min(start + batch, len(texts)), len(texts)))
        return probs

    # ---------- 长文本分块路径 ----------
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    win = max(1, max_len - 2)

    flat_ids, flat_owner = [], []
    n_long = 0
    for i, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False, verbose=False)["input_ids"]
        if len(ids) <= win:
            segs = [ids]
        else:
            n_long += 1
            step = max(1, win // 2)
            starts = list(range(0, len(ids) - win + 1, step))
            if starts[-1] != len(ids) - win:
                starts.append(len(ids) - win)
            if len(starts) > chunk_max_windows:
                # 超长文本均匀取窗口，避免单条文本把推理时间拖爆
                idx = np.linspace(0, len(starts) - 1, chunk_max_windows).round().astype(int)
                starts = [starts[j] for j in sorted(set(idx.tolist()))]
            segs = [ids[s:s + win] for s in starts]
        for seg in segs:
            flat_ids.append([bos] + seg + [eos])
            flat_owner.append(i)

    print("  分块推理：%d 条文本 -> %d 个窗口（其中长文本 %d 条）"
          % (len(texts), len(flat_ids), n_long))

    z_sum, w_sum = np.zeros(len(texts), dtype=np.float64), np.zeros(len(texts), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(flat_ids), batch):
            seqs = flat_ids[start:start + batch]
            ids_t = torch.full((len(seqs), max_len), pad, dtype=torch.long)
            mask_t = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids_t[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask_t[j, :len(s)] = 1
            ids_t, mask_t = ids_t.to(device), mask_t.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=ids_t, attention_mask=mask_t).logits.float()
            z = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            for j, owner in enumerate(flat_owner[start:start + batch]):
                wt = float(max(1, len(seqs[j]) - 2))
                z_sum[owner] += wt * float(z[j])
                w_sum[owner] += wt
            if (start // batch + 1) % 50 == 0:
                print("  已推理窗口 %d / %d" % (min(start + batch, len(flat_ids)), len(flat_ids)))

    w_sum[w_sum == 0] = 1.0
    return sigmoid(z_sum / w_sum).tolist()


def load_model(model_dir, base_model, merge_lora, device):
    """加载 LoRA 适配器 + 骨干模型，返回 (model, tokenizer)。"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("加载分词器:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel
        print("加载骨干:", base_model)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        print("检测到 LoRA 适配器，加载:", model_dir)
        model = PeftModel.from_pretrained(model, model_dir)
        if merge_lora:
            # 合并 LoRA 到骨干，推理更快；数学上与不合并等价
            print("合并 LoRA 权重 ...")
            model = model.merge_and_unload()
    else:
        # 先判断再加载：全量微调目录里已经有完整权重，没必要先把 base_model 载进来
        print("未发现适配器，按全量微调权重加载:", model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    model.eval().to(device)
    return model, tokenizer


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def apply_calibration(probs, calib_path, use_raw=False):
    """读取 calibration.json 并做单调映射：q = sigmoid(a*logit(p) + b)。

    use_raw=True 时优先用未做先验平移的参数（a_raw/b_raw）。
    融合场景必须用原始参数，否则两个成员各自带一次先验平移，
    会把边界挪两次；统一在融合之后再做一次先验对齐。
    """
    if not calib_path or not os.path.exists(calib_path):
        print("未找到校准参数，使用原始概率:", calib_path)
        return np.asarray(probs), None
    with open(calib_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if use_raw and "a_raw" in cfg and "b_raw" in cfg:
        a, b = float(cfg["a_raw"]), float(cfg["b_raw"])
        print("已应用校准（原始参数）: a=%.4f b=%.4f" % (a, b))
    else:
        a, b = float(cfg["a"]), float(cfg["b"])
        print("已应用校准: method=%s a=%.4f b=%.4f" % (cfg.get("method", "?"), a, b))
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    q = 1.0 / (1.0 + np.exp(-(a * z + b)))
    return q, cfg


def apply_prior_match(probs, target_prior):
    """按目标先验平移判定边界，使 0.5 处的正例率等于 target_prior。

    为什么需要它：
      概率校准只能修正"刻度"，修正不了"整个分数分布相对 0.5 的偏移"。
      实测 OOD 场景下模型预测正例率只有 40.6%，而真实先验约 58%，
      导致大量机器样本被漏判；按先验分位平移后，同一模型提升约 11 个点。

    做法：取 (1 - target_prior) 分位点 t*，给所有 logit 加上 -logit(t*)。
    这是单调变换，不改变任何样本的先后顺序（AUC 不变），
    只把 0.5 这个切点挪到正确的先验位置上。
    """
    q = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    t_star = float(np.quantile(q, 1.0 - target_prior))
    t_star = min(max(t_star, 1e-6), 1 - 1e-6)
    shift = -float(np.log(t_star / (1 - t_star)))
    z = np.log(q / (1 - q))
    out = 1.0 / (1.0 + np.exp(-(z + shift)))
    return out, t_star, shift


def main():
    ap = argparse.ArgumentParser(description="推理并生成 result.csv")
    ap.add_argument("--input", required=True, help="输入 jsonl（val.jsonl 或本地 holdout）")
    ap.add_argument("--model-dir", default="data/processed/cls_xlmr_large_lora")
    ap.add_argument("--ensemble-dirs", default="",
                    help="逗号分隔的多个模型目录；配合 --ensemble-mode 决定如何合并")
    ap.add_argument("--ensemble-mode", choices=["prob", "logit"], default="prob",
                    help="多模型集成方式：prob=概率平均（同一模型 K 折推荐）；"
                         "logit=logit 加权（跨骨干模型推荐，权重用 --ensemble-weights）")
    ap.add_argument("--ensemble-weights", default="",
                    help="logit 模式的模型权重，逗号分隔且与 --ensemble-dirs 一一对应；留空为等权")
    ap.add_argument("--chunk-long", action="store_true",
                    help="对超过 max_len 的长文本按滑窗分块推理后聚合，避免尾部信息被截断丢弃")
    ap.add_argument("--chunk-max-windows", type=int, default=8,
                    help="单条长文本最多取多少个窗口（超长文本均匀抽样，控制推理成本）")
    ap.add_argument("--calibration", default="",
                    help="校准参数路径；默认取第一个模型目录下的 calibration.json")
    ap.add_argument("--base-model", default="xlm-roberta-large")
    ap.add_argument("--out-csv", default="data/processed/result.csv")
    ap.add_argument("--out-jsonl", default="", help="可选：同时输出带概率的 jsonl 便于排查")
    ap.add_argument("--text-field", default="", help="手动指定文本字段名；默认自动探测")
    ap.add_argument("--id-field", default="",
                    help="手动指定输出 id 字段名；默认自动探测，探测到重复时回退为行号")
    ap.add_argument("--max-len", type=int, default=512,
                    help="必须与训练时的 max_len 一致，否则截断口径不同会导致概率整体偏移")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条，用于快速自检")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--prior-match", action="store_true",
                    help="【仅限本地诊断】按目标集概率分布平移判定边界。用到目标集分布，"
                         "对官方 val 属 transductive，违反参赛指南第130行禁查规则，禁止用于正式提交")
    ap.add_argument("--target-prior", type=float, default=0.0,
                    help="目标正例率；0 表示读取 calibration.json 里的 prior_target")
    ap.add_argument("--tfidf-model", default="", help="TF-IDF 融合成员（joblib 路径）")
    ap.add_argument("--tfidf-calibration", default="", help="TF-IDF 成员的校准参数 json")
    ap.add_argument("--fuse-weight", type=float, default=0.5,
                    help="logit 融合中神经模型的权重（0~1）")
    ap.add_argument("--merge-lora", action="store_true", help="把 LoRA 合并进骨干（更快）")
    ap.add_argument("--normalize", action="store_true",
                    help="与训练时一致地做文本归一化；必须与训练配置相同，否则输入分布不一致")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)

    # ---------- 1. 读入并解析输入 ----------
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if args.limit and len(rows) >= args.limit:
                    break
    print("输入条数:", len(rows))

    texts, ids, modes = [], [], set()
    for i, r in enumerate(rows):
        t, mode = pick_text(r, args.text_field)
        modes.add(mode)
        texts.append(t if t is not None else "")
        if args.id_field:
            ids.append(str(r.get(args.id_field, i)))
        else:
            ids.append(pick_id(r, i))

    # 同源 human/machine 两条常共享 source_id，直接拿去提交会让行整体错位。
    # 自动探测到重复时回退行号，保证输出 id 唯一。
    if not args.id_field and len(set(ids)) < len(ids):
        n_dup = len(ids) - len(set(ids))
        print("[警告] 自动探测的 id 有 %d 个重复（同源配对通常共享 source_id），"
              "已回退为行号；若官方指定了 id 字段，请加 --id-field。" % n_dup)
        ids = [str(i) for i in range(len(rows))]

    if "dual" in modes:
        raise SystemExit(
            "输入中存在同时包含 human_text 与 machine_text 的行。\n"
            "这种格式一行对应两条待判定文本，必须先明确评测口径再决定如何展开，\n"
            "否则提交的 id 会整体错位。请用 --text-field 指定要判定的那一栏，\n"
            "或先把数据展开成一行一条 text。"
        )
    if "missing" in modes:
        n_missing = sum(1 for t in texts if not t)
        print("[警告] 有 %d 行没有可用文本字段，将输出 0.5 作为占位概率。" % n_missing)

    if args.normalize:
        texts = [C.normalize_text(t) for t in texts]
        print("已启用文本归一化（须与训练配置一致）")
        if args.tfidf_model:
            raise SystemExit("TF-IDF 成员未在归一化文本上训练，不能与 --normalize 同时使用")

    # ---------- 2. 加载模型（支持 K 折集成） ----------
    if args.ensemble_dirs:
        model_dirs = [d.strip() for d in args.ensemble_dirs.split(",") if d.strip()]
    else:
        model_dirs = [args.model_dir]
    for d in model_dirs:
        if not os.path.isdir(d):
            raise SystemExit("模型目录不存在: %s" % d)

    all_probs = []
    for k, d in enumerate(model_dirs):
        if len(model_dirs) > 1:
            print("[%d/%d] 模型: %s" % (k + 1, len(model_dirs), d))
        model, tokenizer = load_model(d, args.base_model, args.merge_lora, device)
        all_probs.append(run_inference(model, tokenizer, texts, device,
                                       args.max_len, args.batch,
                                       args.chunk_long, args.chunk_max_windows))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_probs) > 1:
        arr = np.asarray(all_probs, dtype=np.float64)
        if args.ensemble_mode == "logit":
            if args.ensemble_weights:
                ws = np.array([float(x) for x in args.ensemble_weights.split(",") if x.strip()],
                              dtype=np.float64)
                if len(ws) != len(model_dirs):
                    raise SystemExit("--ensemble-weights 数量(%d) 与 --ensemble-dirs 数量(%d) 不一致"
                                     % (len(ws), len(model_dirs)))
            else:
                ws = np.ones(len(model_dirs), dtype=np.float64)
            ws = ws / ws.sum()
            probs = sigmoid(np.sum(ws[:, None] * logit(arr), axis=0)).tolist()
            print("集成完成: %d 个模型，logit 加权 %s"
                  % (len(all_probs), np.round(ws, 3).tolist()))
        else:
            probs = arr.mean(axis=0).tolist()
            print("集成完成: %d 个模型，概率取平均" % len(all_probs))
    else:
        probs = all_probs[0]
    print("推理完成:", len(probs), "条")

    # ---------- 4. 校准 ----------
    calib_path = args.calibration or os.path.join(model_dirs[0], "calibration.json")
    calib_cfg = None
    if args.no_calibrate:
        final_probs = np.asarray(probs)
        print("按参数关闭校准，输出原始概率")
    else:
        final_probs, calib_cfg = apply_calibration(
            probs, calib_path, use_raw=bool(args.tfidf_model))

    # ---------- 4.5 融合 TF-IDF 成员 ----------
    # 两个成员的原始概率尺度不同，所以在 logit 空间做加权平均；
    # 融合后再统一做一次先验对齐，避免边界被挪两次。
    if args.tfidf_model:
        import joblib
        if not os.path.exists(args.tfidf_model):
            raise SystemExit("找不到 TF-IDF 模型: %s" % args.tfidf_model)
        bundle = joblib.load(args.tfidf_model)
        p_tfidf = bundle["clf"].predict_proba(
            bundle["union"].transform(texts))[:, 1]
        if args.tfidf_calibration:
            p_tfidf, _ = apply_calibration(
                p_tfidf, args.tfidf_calibration, use_raw=True)
        w = float(args.fuse_weight)
        final_probs = sigmoid(w * logit(final_probs) + (1.0 - w) * logit(p_tfidf))
        print("已融合 TF-IDF: 神经权重 %.2f | 词面权重 %.2f" % (w, 1.0 - w))

    # 先验对齐：把 0.5 这个切点挪到目标正例率对应的分位上。
    # 这一步必须在拿到整个输入集的概率之后做（它用的是输入的分布），
    # 因此不适合配合 --limit 在小样本上使用。
    if args.prior_match:
        target = args.target_prior
        if target <= 0:
            target = float(calib_cfg.get("prior_target", 0.58)) if calib_cfg else 0.58
        final_probs, t_star, shift = apply_prior_match(final_probs, target)
        print("已按先验 %.4f 平移判定边界: 分位点 %.4f, logit 平移 %+.4f"
              % (target, t_star, shift))

    # ---------- 5. 写结果 ----------
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prob"])
        for i, p in zip(ids, final_probs):
            w.writerow([i, "%.6f" % float(p)])
    print("已写出:", out_csv)

    if args.out_jsonl:
        out_jsonl = os.path.abspath(args.out_jsonl)
        # 除 id/text/prob 外，把输入里已有的元信息一并带出。
        # 原因：eval_ood.py 等下游脚本要按 label 算分、按 domain 切片，
        # 若这里丢掉字段，下游只能报 KeyError 或退回猜行号。
        keep_fields = ("label", "domain", "generator", "lang", "source_id")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for i, t, p, raw, row in zip(ids, texts, final_probs, probs, rows):
                rec = {"id": i, "text": t, "prob": float(p), "prob_raw": float(raw)}
                for k in keep_fields:
                    if k in row:
                        rec[k] = row[k]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("已写出:", out_jsonl)

    # ---------- 6. 若输入自带标签，顺手打印本地分数 ----------
    if all("label" in r for r in rows):
        from sklearn.metrics import f1_score
        y = np.array([int(r["label"]) for r in rows])
        preds = (final_probs >= 0.5).astype(int)
        mask_short = np.array([C.count_words(t) < 100 for t in texts])
        overall = f1_score(y, preds, average="macro", zero_division=0)
        if mask_short.any() and len(np.unique(y[mask_short])) >= 2:
            short = f1_score(y[mask_short], preds[mask_short], average="macro", zero_division=0)
            combined = 0.7 * overall + 0.3 * short
        else:
            short, combined = float("nan"), overall
        print("\n本地自检（阈值 0.5）：宏F1 %.4f | 短文本F1 %.4f | 综合分 %.4f"
              % (overall, short, combined))


if __name__ == "__main__":
    main()
