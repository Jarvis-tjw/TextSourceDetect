# -*- coding: utf-8 -*-
"""
租卡版训练（train_gpu.py）
==========================
面向 24GB 及以上显存的训练脚本。和本机的 train_cls.py 分开维护，原因：
  1) train_cls.py 为 8GB 显存做了大量妥协（LoRA、max_len 256、固定学习率），
     改它会影响已经验收过的提交链路；
  2) 租卡环境通常是 Linux，需要支持全量微调、512 长度、warmup+cosine，
     以及按 group 的 K 折交叉验证（OOF 用于融合与校准）。

关键设计：
  - 按 group 而不是按行切折：同一篇源文章的人和机器版本必须落在同一折，
    否则 OOF 会被同源泄漏抬高。
  - 每折单独保存模型与预测，机器中断只损失一折，可续跑。
  - 监控指标用 AUC 选最佳 epoch（阈值无关，避免用标签调阈值造成乐观偏差）。

用法：
  # 单次全量微调（用 holdout 监控）
  python scripts/train_gpu.py --mode full --outdir data/processed/cls_full

  # 5 折中的第 0 折
  python scripts/train_gpu.py --mode full --cv-folds 5 --cv-index 0 \
      --outdir data/processed/cls_full_5fold

  # 租卡后的环境冒烟（几分钟）
  python scripts/train_gpu.py --mode full --max-train 2000 --max-len 256 \
      --epochs 1 --outdir data/processed/smoke_gpu
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def parse_args():
    ap = argparse.ArgumentParser(description="租卡版：全量微调 / LoRA + K 折")
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--train", default=os.path.join(C.PROCESSED_DIR, "train.jsonl"))
    ap.add_argument("--holdout", default=os.path.join(C.PROCESSED_DIR, "holdout.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(C.PROCESSED_DIR, "cls_full"))
    ap.add_argument("--mode", choices=["full", "lora"], default="full")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.0, help="0 表示按模式自动选：full=2e-5, lora=1e-4")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="梯度检查点，省显存；默认开启，可用 --no-grad-ckpt 关闭")
    ap.add_argument("--cv-folds", type=int, default=0, help="0=不切折；>1=按 group 切 K 折")
    ap.add_argument("--cv-index", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="0=全量；冒烟时可设小值")
    ap.add_argument("--max-val", type=int, default=0, help="0=全量；冒烟时可只验证前 N 条")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", default="query,value")
    # ---- 同源配对损失（域不变性核心）：同一 group 内 machine 的分数必须高于 human ----
    ap.add_argument("--pair-loss-weight", type=float, default=0.0,
                    help="配对 margin loss 权重，0=关闭。建议 0.3 起步")
    ap.add_argument("--pair-margin", type=float, default=1.0,
                    help="要求 logit(machine) - logit(human) 超过该间隔（logit 空间）")
    ap.add_argument("--pair-min-jaccard", type=float, default=0.05,
                    help="配对文本 char-3gram Jaccard 下限，过滤不同题的伪配对")
    ap.add_argument("--pair-per-step", type=int, default=2,
                    help="每个优化步额外采样的配对数（每对 2 段文本）")
    ap.add_argument("--max-pairs", type=int, default=200000, help="候选配对上限，控制内存")
    ap.add_argument("--extra-val", default="",
                    help="额外评估集，逗号分隔的 jsonl 路径，只打印指标不影响选模型")
    ap.add_argument("--max-steps", type=int, default=0, help="0=完整 epoch；>0 只跑前 N 步（冒烟）")
    ap.add_argument("--subsample", type=int, default=0,
                    help="0=全量；>0 按 group 整组抽约 N 条（A/B 加速，保证配对不被拆散）")
    ap.add_argument("--normalize", action="store_true",
                    help="训练/验证/额外评估统一做文本归一化（URL/公式/引用/代码块→占位符）")
    ap.add_argument("--domain-balance", type=float, default=0.0,
                    help="域平衡采样强度 α：0=关闭；0.5=温和；1.0=每域近似等量")
    # ---- 生成器辅助多任务：让共享表示学会"不同模型各自的写作习惯"，为复赛归因铺路 ----
    ap.add_argument("--gen-loss-weight", type=float, default=0.0,
                    help="生成器辅助头损失权重，0=关闭；建议 0.2 起步")
    ap.add_argument("--gen-head-lr-mult", type=float, default=10.0,
                    help="生成器头学习率倍数：随机初始化的头用主学习率会学不动")
    # ---- 域对抗（DANN）：用梯度反转逼编码器丢掉"这段文本属于哪个域"的信息 ----
    ap.add_argument("--dann-weight", type=float, default=0.0,
                    help="域对抗最终强度 λ，0=关闭。建议 0.1 起步")
    ap.add_argument("--dann-warm", type=float, default=0.3,
                    help="λ 从 0 线性升到 dann-weight 所用的训练进度比例")
    ap.add_argument("--dann-head-lr-mult", type=float, default=10.0,
                    help="域分类头学习率倍数：新初始化的头用主学习率会学不动，对抗信号等于没有")
    args = ap.parse_args()
    if args.lr <= 0:
        args.lr = 2e-5 if args.mode == "full" else 1e-4
    return args


def set_seed(seed):
    """固定随机源，并要求 cuDNN 用确定性算法（租卡环境换机器也能复现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(path, max_n=0, normalize=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if normalize:
                r["text"] = C.normalize_text(r.get("text", ""))
            rows.append(r)
            if max_n and len(rows) >= max_n:
                break
    return rows


def group_key(r, idx):
    """优先用 group；缺失时退回 source_id；再缺就用行号，保证每行有稳定归属。"""
    return str(r.get("group") or r.get("source_id") or ("__row%d" % idx))


def fold_of(key, k):
    """用 md5 做稳定分折：不依赖 Python 的随机哈希盐，跨机器结果一致。"""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return h % k


def char_grams(text, n=3):
    """char n-gram 集合。用字符而不是词，才能同时覆盖中/阿/俄等无空格语言。"""
    t = " ".join(text.split())
    return {t[i:i + n] for i in range(0, max(0, len(t) - n + 1))}


def build_pairs(tr_rows, min_jaccard=0.05, max_pairs=200000):
    """构造同源配对索引：同 group、一 human 一 machine、字面重叠达阈值。

    为什么按 group 逐组处理：gram 集合只在一组内短暂存活，峰值内存可控；
    若给全部 11 万条文本建常驻 gram 表，内存会翻到 GB 级。

    返回 [(pos_h, pos_m), ...]，pos 是 tr_rows 中的位置（不是原始行号）。
    """
    by_group = {}
    for pos, (_, r) in enumerate(tr_rows):
        g = group_key(r, pos)
        by_group.setdefault(g, []).append(pos)

    pairs = []
    for positions in by_group.values():
        if len(positions) < 2:
            continue
        hs = [p for p in positions if int(tr_rows[p][1]["label"]) == 0]
        ms = [p for p in positions if int(tr_rows[p][1]["label"]) == 1]
        if not hs or not ms:
            continue
        h_grams = [(p, char_grams(tr_rows[p][1]["text"])) for p in hs]
        m_grams = [(p, char_grams(tr_rows[p][1]["text"])) for p in ms]
        for ph, gh in h_grams:
            for pm, gm in m_grams:
                inter = len(gh & gm)
                if inter == 0:
                    continue
                if inter / max(1, len(gh | gm)) >= min_jaccard:
                    pairs.append((ph, pm))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


class GradReverse(torch.autograd.Function):
    """梯度反转层：前向直通，反向把梯度乘 -λ。

    域分类头希望"看得出这是哪个域"，反转后编码器收到的梯度方向是
    "让自己更看不出这是哪个域"，从而把域信息从表征里挤出去。
    """
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class TextDS(Dataset):
    """取 (text, label, domain_id, gen_id)。
    domain_id / gen_id 在无映射时统一给 -100（交叉熵忽略）。
    gen_id 只对 machine 文本有意义：human 文本没有生成器，一律 -100。
    """

    def __init__(self, rows, dom_map=None, gen_map=None):
        self.rows = rows
        self.doms = ([dom_map.get(r.get("domain"), -100) for r in rows]
                     if dom_map else None)
        self.gens = ([gen_map.get(r.get("generator"), -100)
                      if int(r["label"]) == 1 else -100 for r in rows]
                     if gen_map else None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (r["text"], int(r["label"]),
                (self.doms[i] if self.doms is not None else -100),
                (self.gens[i] if self.gens is not None else -100))


class Collator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        domains = [b[2] for b in batch]
        gens = [b[3] for b in batch]
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_len,
            padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["domains"] = torch.tensor(domains, dtype=torch.long)
        enc["gens"] = torch.tensor(gens, dtype=torch.long)
        return enc


def forward_classifier(model, domain_head, gen_head, batch, lambd):
    """一次编码前向，同时产出主任务 logits、域对抗 logits、生成器 logits。

    为什么不直接用 model(...)：那样拿不到编码器输出，为了接辅助头就得再跑一遍编码。
    这里手动走 base_model -> classifier，复用同一份编码结果，几乎零额外算力。
    """
    if domain_head is None and gen_head is None:
        return model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"]).logits.float(), None, None
    enc = model.base_model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
    seq = enc.last_hidden_state
    logits = model.classifier(seq).float()
    mask = batch["attention_mask"].unsqueeze(-1).to(seq.dtype)
    pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    d_logits = (domain_head(GradReverse.apply(pooled.float(), lambd))
                if domain_head is not None else None)
    g_logits = gen_head(pooled.float()) if gen_head is not None else None
    return logits, d_logits, g_logits


def build_model(args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0.1, target_modules=targets, bias="none"))
        model.print_trainable_parameters()

    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if args.mode == "lora" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model.to(device), tokenizer


def predict_probs(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return probs


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


def best_threshold(y, p, texts):
    best_th, best = 0.5, -1.0
    for th in np.linspace(0.01, 0.99, 197):
        s = score(y, p, texts, float(th))["combined"]
        if s > best:
            best_th, best = float(th), float(s)
    return best_th, best


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备: %s | 模式: %s | max_len: %d | batch: %d x accum %d"
          % (device, args.mode, args.max_len, args.batch, args.grad_accum), flush=True)
    if device == "cuda":
        print("GPU: %s | 显存 %.1f GB"
              % (torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

    train_rows = load_rows(args.train, args.max_train, args.normalize)
    holdout_rows = load_rows(args.holdout, args.max_val, args.normalize)
    if args.normalize:
        print("已启用文本归一化（训练/验证/额外评估一致）", flush=True)
    print("训练集: %d | holdout: %d" % (len(train_rows), len(holdout_rows)), flush=True)

    # 按 group 整组抽样子采样：只抽单条会把同源配对拆散，A/B 必须成组抽样
    if args.subsample and args.subsample < len(train_rows):
        by_g = {}
        for i, r in enumerate(train_rows):
            by_g.setdefault(group_key(r, i), []).append(r)
        keys = list(by_g)
        random.Random(args.seed).shuffle(keys)
        picked, n = [], 0
        for k in keys:
            picked.extend(by_g[k])
            n += len(by_g[k])
            if n >= args.subsample:
                break
        train_rows = picked
        print("按 group 子采样: %d 条（目标 %d，组完整）" % (len(train_rows), args.subsample),
              flush=True)

    # ---------- 切分：K 折用训练集内部切；否则用固定 holdout 监控 ----------
    if args.cv_folds and args.cv_folds > 1:
        if not (0 <= args.cv_index < args.cv_folds):
            raise SystemExit("--cv-index 必须在 [0, %d) 内" % args.cv_folds)
        tr_rows, va_rows = [], []
        for i, r in enumerate(train_rows):
            if fold_of(group_key(r, i), args.cv_folds) == args.cv_index:
                va_rows.append((i, r))
            else:
                tr_rows.append((i, r))
        if not tr_rows or not va_rows:
            raise SystemExit("某一折为空，检查 --cv-folds/--cv-index")
        y_va = [int(r["label"]) for _, r in va_rows]
        if len(set(y_va)) < 2:
            raise SystemExit("验证折只有单一类别，换一个 --cv-index 或调整分折种子")
        print("K 折 %d/%d | 训练 %d | 验证 %d"
              % (args.cv_index + 1, args.cv_folds, len(tr_rows), len(va_rows)), flush=True)
        pred_name = "pred_fold%d.jsonl" % args.cv_index
        monitor_name = "fold%d" % args.cv_index
    else:
        tr_rows = list(enumerate(train_rows))
        va_rows = list(enumerate(holdout_rows))
        pred_name = "pred_holdout.jsonl"
        monitor_name = "holdout"

    # ---------- 域标签映射（unknown 视为无域，参与主任务但不参与域对抗） ----------
    dom_map = {}
    if args.dann_weight > 0:
        doms = sorted({r.get("domain") for _, r in tr_rows
                       if r.get("domain") not in (None, "", "unknown")})
        dom_map = {d: i for i, d in enumerate(doms)}
        print("DANN: %d 个域 %s" % (len(doms), doms), flush=True)

    # ---------- 生成器标签映射（只对 machine 样本有效；unknown 不参与） ----------
    gen_map = {}
    if args.gen_loss_weight > 0:
        gens = sorted({r.get("generator") for _, r in tr_rows
                       if int(r["label"]) == 1
                       and r.get("generator") not in (None, "", "unknown")})
        gen_map = {g: i for i, g in enumerate(gens)}
        print("生成器辅助头: %d 类 %s" % (len(gens), gens), flush=True)

    train_ds = TextDS([r for _, r in tr_rows], dom_map, gen_map)
    val_ds = TextDS([r for _, r in va_rows], dom_map, gen_map)

    # ---------- 同源配对（仅训练折内构造，绝不跨折，避免泄漏） ----------
    pair_list = []
    if args.pair_loss_weight > 0:
        t_pair = time.time()
        pair_list = build_pairs(tr_rows, args.pair_min_jaccard, args.max_pairs)
        print("同源配对: %d 对（Jaccard>=%.2f，用时 %.0fs）"
              % (len(pair_list), args.pair_min_jaccard, time.time() - t_pair), flush=True)
        if not pair_list:
            raise SystemExit("没有可用配对，检查 group 字段或调低 --pair-min-jaccard")

    # 额外评估集（只打印，不参与选模型）：用于同时看同域是否回退
    extra_sets = []
    for p in [x.strip() for x in args.extra_val.split(",") if x.strip()]:
        rows_ex = load_rows(p, args.max_val, args.normalize)
        extra_sets.append((p, rows_ex, TextDS(rows_ex)))
        print("额外评估集: %s (%d 条)" % (p, len(rows_ex)), flush=True)

    model, tokenizer = build_model(args, device)
    collate = Collator(tokenizer, args.max_len)

    # 域分类头只在训练期存在，推理时丢弃（不参与保存），这是 DANN 的关键前提
    domain_head = None
    if dom_map:
        if args.mode == "lora":
            raise SystemExit("DANN 需要全量微调（--mode full），LoRA 冻结骨干时无法做域对抗")
        hidden = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
        domain_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, len(dom_map))).to(device)
        print("DANN 域头: %d -> 256 -> %d，λ=%.2f，warm=%.0f%%"
              % (hidden, len(dom_map), args.dann_weight, args.dann_warm * 100), flush=True)

    # 生成器辅助头同样只在训练期存在，推理时丢弃，不影响主任务前向
    gen_head = None
    if gen_map:
        if args.mode == "lora":
            raise SystemExit("生成器辅助头需要全量微调（--mode full），LoRA 下骨干冻结、辅助信号无法回传")
        hidden = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
        gen_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, len(gen_map))).to(device)
        print("生成器头: %d -> 256 -> %d，权重 %.2f"
              % (hidden, len(gen_map), args.gen_loss_weight), flush=True)
    # 域平衡采样：按 1/n_domain^α 给样本加权，避免大域主导梯度，
    # 让小域（baike/reddit/阿拉伯语等）在训练中有足够曝光。
    sampler = None
    if args.domain_balance > 0:
        from collections import Counter
        counts = Counter(r.get("domain", "unknown") for _, r in tr_rows)
        wts = [1.0 / (counts[r.get("domain", "unknown")] ** args.domain_balance)
               for _, r in tr_rows]
        sampler = WeightedRandomSampler(wts, num_samples=len(wts), replacement=True)
        print("域平衡采样 α=%.2f | %d 个域 | 最大域 %d 条（占 %.1f%%）"
              % (args.domain_balance, len(counts), max(counts.values()),
                 100.0 * max(counts.values()) / len(wts)), flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)

    # ---------- 类别加权（machine:human 约 1.39:1，轻度不平衡） ----------
    labels_tr = np.array([int(r["label"]) for _, r in tr_rows])
    n_machine = int(labels_tr.sum())
    n_human = len(labels_tr) - n_machine
    w = torch.tensor([len(labels_tr) / max(1, n_human),
                      len(labels_tr) / max(1, n_machine)], dtype=torch.float32)
    w = (w / w.mean()).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": args.lr}]
    if domain_head is not None:
        # 域头是随机初始化的，和预训练骨干共用 2e-5 会几乎学不动（实测停在随机水平），
        # 单独给它更高学习率，保证对抗信号真实存在，否则 A/B 会是假阴性。
        groups.append({"params": list(domain_head.parameters()),
                       "lr": args.lr * args.dann_head_lr_mult})
        params = params + list(domain_head.parameters())
    if gen_head is not None:
        groups.append({"params": list(gen_head.parameters()),
                       "lr": args.lr * args.gen_head_lr_mult})
        params = params + list(gen_head.parameters())
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch / args.grad_accum)
    total_steps = max(1, steps_per_epoch * int(math.ceil(args.epochs)))
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps)

    best_auc, best_state, best_th, best_metrics = -1.0, None, 0.5, {}
    t0 = time.time()
    gstep = 0
    ramp_steps = max(1, int(total_steps * args.dann_warm))
    for epoch in range(int(math.ceil(args.epochs))):
        model.train()
        running, running_pair, running_dom, running_gen = 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # DANN 的 λ 从 0 线性升到设定值，避免训练初期域对抗把还没成型的表征打散
            lambd = args.dann_weight * min(1.0, gstep / ramp_steps)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits, d_logits, g_logits = forward_classifier(
                    model, domain_head, gen_head, batch, lambd)
                loss = loss_fn(logits, batch["labels"]) / args.grad_accum
                dom_loss_val = 0.0
                if d_logits is not None:
                    d_loss = torch.nn.functional.cross_entropy(
                        d_logits, batch["domains"], ignore_index=-100)
                    loss = loss + d_loss / args.grad_accum
                    dom_loss_val = float(d_loss.item())

                # 生成器辅助损失：只有 machine 样本有标签，human 全被 ignore。
                # 一个 batch 全是 human 时交叉熵会是 nan，必须先判断再算。
                gen_loss_val = 0.0
                if g_logits is not None and (batch["gens"] != -100).any():
                    g_loss = torch.nn.functional.cross_entropy(
                        g_logits, batch["gens"], ignore_index=-100)
                    loss = loss + args.gen_loss_weight * g_loss / args.grad_accum
                    gen_loss_val = float(g_loss.item())

                # 同源配对 margin loss：同题同域下，machine 的分数必须明显高于 human。
                # 域/主题/长度线索在配对内部相互抵消，模型只能去学"谁写的"。
                pair_loss_val = 0.0
                if pair_list:
                    sampled = [random.choice(pair_list) for _ in range(args.pair_per_step)]
                    items = []
                    for ph, pm in sampled:
                        items.append((tr_rows[pm][1]["text"], 1, -100, -100))
                        items.append((tr_rows[ph][1]["text"], 0, -100, -100))
                    pb = {k: v.to(device) for k, v in collate(items).items()}
                    p_logits = model(input_ids=pb["input_ids"],
                                     attention_mask=pb["attention_mask"]).logits.float()
                    # items 顺序为 [machine, human, machine, human, ...]
                    # 用 softplus 而不是 relu：hinge 一旦拉开到 margin 就梯度归零，
                    # 长时间训练后配对损失会饱和成 0；softplus 平滑、可持续施压。
                    pair_loss = torch.nn.functional.softplus(
                        args.pair_margin - (p_logits[0::2, 1] - p_logits[1::2, 1])).mean()
                    loss = loss + args.pair_loss_weight * pair_loss / args.grad_accum
                    pair_loss_val = float(pair_loss.item())
            loss.backward()
            running += loss.item()
            running_pair += pair_loss_val
            running_dom += dom_loss_val
            running_gen += gen_loss_val
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if (step // args.grad_accum + 1) % 50 == 0:
                    # running 是这 50 个优化步内所有微批次的 loss 之和；
                    # 除以 50 得到"每个优化步的平均 loss"，量级直观。
                    msg = "  epoch %d step %d loss %.4f" % (epoch + 1, step + 1, running / 50.0)
                    if pair_list:
                        msg += " pair %.4f" % (running_pair / 50.0)
                    if domain_head is not None:
                        msg += " dom %.3f λ%.3f" % (running_dom / 50.0, lambd)
                    if gen_head is not None:
                        msg += " gen %.3f" % (running_gen / 50.0)
                    msg += " lr %.2e 已用 %.0fs" % (sched.get_last_lr()[0], time.time() - t0)
                    print(msg, flush=True)
                    running, running_pair, running_dom, running_gen = 0.0, 0.0, 0.0, 0.0
            if args.max_steps and step + 1 >= args.max_steps:
                print("达到 --max-steps=%d，提前结束本 epoch" % args.max_steps, flush=True)
                break

        # 最后一个不足 grad_accum 的批次也要更新一次，否则每 epoch 尾部梯度被丢弃
        if (step + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

        # 每个 epoch 结束在验证集上评估；用 AUC 选最佳（阈值无关）
        val_probs = predict_probs(model, val_loader, device)
        y_val = [int(r["label"]) for _, r in va_rows]
        t_val = [r.get("text", "") for _, r in va_rows]
        m = score(y_val, val_probs, t_val, 0.5)
        th, best_comb = best_threshold(y_val, val_probs, t_val)
        print("epoch %d | AUC %.4f | 宏F1@0.5 %.4f | 最优阈值 %.3f -> %.4f | 用时 %.0fs"
              % (epoch + 1, m["auc"], m["macro_f1"], th, best_comb, time.time() - t0),
              flush=True)

        # 额外评估集：看同域有没有被配对损失拖回退（不参与选模型）
        for ep, rows_ex, ds_ex in extra_sets:
            loader_ex = DataLoader(ds_ex, batch_size=args.batch, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True,
                                   collate_fn=collate)
            p_ex = predict_probs(model, loader_ex, device)
            m_ex = score([int(r["label"]) for r in rows_ex], p_ex,
                         [r.get("text", "") for r in rows_ex], 0.5)
            print("    extra %s | AUC %.4f | 宏F1@0.5 %.4f"
                  % (os.path.basename(ep), m_ex["auc"], m_ex["macro_f1"]), flush=True)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_th = th
            best_metrics = {"auc": m["auc"], "macro_f1": m["macro_f1"],
                            "short_f1": m["short_f1"], "best_threshold": th,
                            "best_combined": best_comb, "epoch": epoch + 1}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    save_dir = os.path.join(args.outdir, monitor_name) if args.cv_folds > 1 else args.outdir
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("最佳模型已保存:", save_dir, flush=True)

    # ---------- 用最佳模型重新预测并落盘 ----------
    val_probs = predict_probs(model, val_loader, device)

    # 额外评估集也用"最佳模型"重新推理并落盘：epoch 循环里打印的是各 epoch 当时的状态，
    # 最终产物必须对应被选中的那个模型，否则后续校准/汇总会用到次优权重。
    extra_metrics = {}
    for ep, rows_ex, ds_ex in extra_sets:
        loader_ex = DataLoader(ds_ex, batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate)
        p_ex = predict_probs(model, loader_ex, device)
        m_ex = score([int(r["label"]) for r in rows_ex], p_ex,
                     [r.get("text", "") for r in rows_ex], 0.5)
        name = os.path.splitext(os.path.basename(ep))[0]
        ex_path = os.path.join(args.outdir, "pred_extra_%s.jsonl" % name)
        with open(ex_path, "w", encoding="utf-8") as f:
            for r, p in zip(rows_ex, p_ex):
                rec = {"text": r.get("text", ""), "label": int(r["label"]),
                       "prob": float(p)}
                for k in ("group", "domain", "generator", "lang", "source_id"):
                    if k in r:
                        rec[k] = r[k]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra_metrics[name] = m_ex
        print("    extra 落盘 %s | AUC %.4f | 宏F1@0.5 %.4f"
              % (ex_path, m_ex["auc"], m_ex["macro_f1"]), flush=True)

    pred_path = os.path.join(args.outdir, pred_name)
    with open(pred_path, "w", encoding="utf-8") as f:
        for (orig_id, r), p in zip(va_rows, val_probs):
            rec = {"id": orig_id, "text": r.get("text", ""),
                   "label": int(r["label"]), "prob": float(p)}
            for k in ("group", "domain", "generator", "lang", "source_id"):
                if k in r:
                    rec[k] = r[k]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = dict(best_metrics)
    metrics.update({"pred": pred_path, "model_dir": save_dir, "mode": args.mode,
                    "max_len": args.max_len, "seed": args.seed, "lr": args.lr,
                    "pair_loss_weight": args.pair_loss_weight,
                    "pair_count": len(pair_list),
                    "pair_min_jaccard": args.pair_min_jaccard,
                    "pair_margin": args.pair_margin,
                    "dann_weight": args.dann_weight,
                    "dann_domains": len(dom_map),
                    "normalize": args.normalize,
                    "domain_balance": args.domain_balance,
                    "gen_loss_weight": args.gen_loss_weight,
                    "gen_classes": len(gen_map),
                    "extra": extra_metrics})
    with open(os.path.join(args.outdir, "metrics_%s.json" % monitor_name), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("预测已写出:", pred_path)
    print("最终指标:", json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
