# -*- coding: utf-8 -*-
"""
Phase 2 神经主力训练（train_cls.py）
===================================
在 8GB 显存的机器上，用【large 多语模型 + LoRA】做 human/machine 二分类。

为什么用 LoRA（而不是全量微调）？
  8GB 显存装不下 large 模型全量微调所需的"权重+梯度+优化器状态"。
  LoRA 只训练一小部分低秩适配器参数（冻结骨干网络），显存占用大幅下降，
  在分类任务上性能通常接近全量微调。

为什么用多语模型（mDeBERTa-v3 / XLM-R）？
  实测数据是"英/俄/阿/中…"多语言，单语 BERT 覆盖不了。

关键训练技巧（都针对 8GB 做了取舍）：
  - bf16 混合精度：省显存、更稳（RTX 5060 支持）
  - gradient_checkpointing：用计算换显存
  - 梯度累积：小 batch 也能有等效大 batch
  - 冻结骨干 + 只训 LoRA 与分类头：显存最省
  - 类别加权：machine:human ≈ 1.39:1，做轻度平衡

运行示例（先小样本冒烟测试，确认能跑通）：
  python scripts/train_cls.py --model microsoft/mdeberta-v3-large \
      --lora --lora-targets query_proj,value_proj \
      --max-len 256 --batch 8 --grad-accum 4 --epochs 2 --max-train 2000
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import eval as E


def parse_args():
    ap = argparse.ArgumentParser(description="large 多语模型 + LoRA 二分类训练")
    ap.add_argument("--model", default="microsoft/mdeberta-v3-large",
                    help="HuggingFace 模型名。可换 microsoft/mdeberta-v3-base 或 xlm-roberta-large")
    ap.add_argument("--train", default=os.path.join(C.PROCESSED_DIR, "train.jsonl"))
    ap.add_argument("--holdout", default=os.path.join(C.PROCESSED_DIR, "holdout.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(C.PROCESSED_DIR, "cls_model"))
    ap.add_argument("--max-len", type=int, default=256, help="最大序列长度（8GB 下不宜过大）")
    ap.add_argument("--batch", type=int, default=8, help="每卡 batch size")
    ap.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA 学习率（比全量微调大）")
    ap.add_argument("--lora", action="store_true", help="启用 LoRA（推荐，8GB 必开）")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", default="query_proj,value_proj",
                    help="LoRA 目标模块，逗号分隔。DeBERTa 用 query_proj,value_proj；XLM-R 用 query,value")
    ap.add_argument("--max-train", type=int, default=0, help="0=全量；否则只取这么多条训练样本")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def set_seed(seed):
    """固定所有随机源，保证可复现（参赛硬性要求）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path, max_n=0):
    """读取预处理后的样本，返回 (texts, labels)。"""
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


class TextClsDataset(Dataset):
    """把 (文本, 标签) 包装成 Transformers 训练需要的格式。
    在这里做分词：截断到 max_len，补齐到 batch 内最长。
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_model_and_tokenizer(args):
    """加载分词器与模型；可选套上 LoRA；开启梯度检查点。"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # num_labels=2 -> human(0) / machine(1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=targets,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()  # 打印可训练参数量，直观看到 LoRA 有多省

    # 梯度检查点：用计算换显存，8GB 必备
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # LoRA + 梯度检查点组合时，必须让"输入 embedding"也参与求导，
    # 否则冻结骨干后反向传播到不了适配器，会报 "none of the inputs require grad"。
    if args.lora and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device, "| 模型:", args.model, "| LoRA:", args.lora)

    train_texts, train_labels = load_jsonl(args.train, args.max_train)
    hold_texts, hold_labels = load_jsonl(args.holdout, 0)
    print("训练样本:", len(train_texts), "| holdout 样本:", len(hold_texts))

    model, tokenizer = build_model_and_tokenizer(args)

    train_ds = TextClsDataset(train_texts, train_labels, tokenizer, args.max_len)
    hold_ds = TextClsDataset(hold_texts, hold_labels, tokenizer, args.max_len)

    # ---- 类别权重：machine:human ≈ 1.39:1，轻度加权让两类更平衡 ----
    n = len(train_labels)
    n_machine = sum(train_labels)
    n_human = n - n_machine
    w = torch.tensor([n / max(1, n_human), n / max(1, n_machine)], dtype=torch.float32)
    # 归一化到均值为 1，避免整体缩放学习率
    w = w / w.mean()

    class WeightedTrainer:
        """极简训练循环，避免依赖 HF Trainer 的版本差异；逻辑清晰、易复现。"""
        def __init__(self, model, ds, w, args, device):
            self.model = model
            self.ds = ds
            self.w = w.to(device)
            self.args = args
            self.device = device

        def run(self):
            from torch.utils.data import DataLoader
            model = self.model.to(self.device)
            loader = DataLoader(self.ds, batch_size=self.args.batch, shuffle=True)
            # 只优化"可训练参数"（LoRA 时骨干是冻结的，优化器只含适配器+分类头）
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=self.args.lr)
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.w)
            # bf16 混合精度：省显存、更稳（RTX 5060 支持）；bf16 不需要 GradScaler
            use_bf16 = torch.cuda.is_available()
            model.train()
            global_step = 0
            t_start = time.time()   # 记录起点，便于估算训练速度
            for epoch in range(int(np.ceil(self.args.epochs))):
                for step, batch in enumerate(loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=use_bf16):
                        out = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"])
                        loss = loss_fn(out.logits, batch["labels"]) / self.args.grad_accum
                    loss.backward()
                    if (step + 1) % self.args.grad_accum == 0:
                        opt.step()          # 累积够 grad_accum 步才更新一次
                        opt.zero_grad()
                        global_step += 1
                        if global_step % 50 == 0:
                            print("  step %d loss %.4f | 已用 %.0fs" % (
                                global_step, loss.item() * self.args.grad_accum, time.time() - t_start))
                print("epoch %d done | 本阶段已用 %.0fs" % (epoch + 1, time.time() - t_start))
                # 每个 epoch 存一份适配器，训练漫长时防止意外中断丢结果
                self.model.save_pretrained(os.path.join(self.args.outdir, "epoch%d" % (epoch + 1)))
            return model

    WeightedTrainer(model, train_ds, w, args, device).run()

    # 训练结束保存最终模型（LoRA 适配器）与分词器，供后续在官方 val 上推理/复现
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("模型已保存到:", args.outdir)

    # ---------------- 在 holdout 上预测概率 + 搜阈值 ----------------
    print("预测 holdout ...")
    probs = predict_probs(model, hold_ds, device, args.batch)

    th, metrics = E.search_threshold(hold_labels, probs, hold_texts)
    print("holdout 最优阈值 %.2f -> %s" % (th, json.dumps(metrics, ensure_ascii=False)))

    pred_path = os.path.join(args.outdir, "pred_holdout.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (txt, y, p) in enumerate(zip(hold_texts, hold_labels, probs)):
            f.write(json.dumps({"id": i, "text": txt, "label": y, "prob": float(p)},
                               ensure_ascii=False) + "\n")
    # 保存阈值，供最终提交时复用
    with open(os.path.join(args.outdir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": th, "metrics": metrics, "model": args.model}, f, ensure_ascii=False, indent=2)
    print("预测与阈值已保存到:", args.outdir)


def predict_probs(model, ds, device, batch):
    """对数据集逐批推理，返回"是 machine"的概率列表。"""
    from torch.utils.data import DataLoader
    model.eval().to(device)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    use_bf16 = torch.cuda.is_available()
    probs = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=batch_data["input_ids"],
                               attention_mask=batch_data["attention_mask"]).logits
            # 概率计算统一转 float32，避免 bf16 精度误差影响阈值搜索
            p = torch.softmax(logits.float(), dim=-1)[:, 1]  # 第 1 类是 machine
            probs.extend(p.cpu().tolist())
    return probs


if __name__ == "__main__":
    main()
