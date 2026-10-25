# 大模型生成文本检测（NLP-1 初赛）

面向「大模型生成文本检测与生成器归因」赛题初赛的二分类方案：判定一段文本来自**人类撰写（human）**还是**AI 机器生成（machine）**，并输出预测概率。初赛只评估二分类性能，不涉及生成器归因与片段级溯源。

## 成绩

5 折交叉验证（OOF，140,378 条全量样本，同一源文章的 human/machine 版本严格同折，无同源泄漏）：

| 指标 | 数值 |
|---|---|
| AUC | **0.9979** |
| 宏 F1 @0.5 | 0.9790 |
| 短文本（<100 词）F1 | 0.9848 |
| **综合分 @0.5**（0.7×宏F1 + 0.3×短文本F1） | **0.9807** |
| 校准 + 先验对齐后（独立 30% 验证半集） | 0.9864 |

各折表现：

| 折 | AUC | 宏F1 | 短文本F1 | 最佳 epoch |
|---|---|---|---|---|
| fold0 | 0.9992 | 0.9810 | 0.9857 | 2 |
| fold1 | 0.9989 | 0.9779 | 0.9830 | 2 |
| fold2 | 0.9986 | 0.9769 | 0.9821 | 2 |
| fold3 | 0.9981 | 0.9782 | 0.9853 | 2 |
| fold4 | 0.9993 | 0.9811 | 0.9875 | 2 |

> 上述分数均为**同域**验证结果。AUC 已接近饱和，不能外推到官方测试集；跨域泛化是主要风险点。

## 方法

**主模型：XLM-RoBERTa-large 全量微调**

- `XLMRobertaForSequenceClassification`，`num_labels=2`（human=0 / machine=1）
- `max_len=512`，batch 16，梯度累积 1，2 epoch，lr 2e-5，cosine + 6% warmup，gradient checkpointing，bf16 autocast
- 固定随机种子（fold i 使用 `seed = 42 + i`），保证可复现

**5 折切分：按 group 而非按行**

同一篇源文章的人类版本与机器版本共享 `source_id`/`group`，若按行随机切折会造成同源泄漏、把 OOF 分数虚高。本项目用 `md5(group) % 5` 稳定分折，跨折无泄漏、且分折结果与随机种子无关。

**融合成员：TF-IDF + LogisticRegression**

词级 1-2gram + `char_wb` 3-5gram，`C=4.0`，`class_weight=balanced`。与神经模型在 logit 空间按权重融合（默认等权，跨骨干集成时可调）。

**校准与阈值对齐**

1. Platt 概率校准（在训练侧 OOF 上拟合参数 `a`、`T`）
2. 先验对齐：把预测正例率平移匹配训练集先验（≈58.05%）
3. 全过程**只用训练侧数据**，不触碰目标集分布（见下方合规说明）

**长文本推理：滑窗分块聚合**

约 28% 的样本超过 512 token（wikihow 域高达 61%），单次截断会丢弃尾部信息。提交推理开启 `--chunk-long`：按 50% 重叠滑窗切分，超长文本均匀抽样最多 N 个窗口，在 logit 空间按窗口长度加权平均。短文本路径与原来完全一致，不改变既有行为。

## 目录结构

```
scripts/
  common.py            公共工具：文本归一化、词数统计、路径常量
  preprocess.py        原始 data.jsonl → train.jsonl / holdout.jsonl
  analyze_data.py      数据分布分析（语种、领域、长度、生成器）
  baseline_tfidf.py    TF-IDF 基线模型训练与预测
  train_gpu.py         租卡训练主脚本：全量微调 / LoRA + 按 group 的 K 折
  train_cls.py         本机低显存训练脚本（LoRA、max_len 256）
  merge_cv.py          合并 K 折预测，产出干净 OOF 与汇总
  calibrate.py         在 OOF 上拟合 Platt 校准 + 先验对齐参数
  predict.py           最终推理入口，支持多模型集成与长文本分块
  make_ood_split.py    构造领域留出（OOD）划分
  eval.py              基础评估
  eval_cross_domain.py 跨域评估
  eval_ood.py          OOD 评估
  eval_ood_bench.py    OOD 压力测试
  extract_graybox.py   灰盒特征提取（PPL / GLTR）
train_rented.sh        租卡一键训练脚本（冒烟 / 单模型 / 5 折）
requirements.txt       固定版本的依赖清单
Dockerfile             运行环境镜像
repro.ps1              Windows 本地复现脚本
大模型检测与归因方案设计.md  完整方案设计文档
```

产物目录（不入库）：

```
data/processed/cls_full_5fold/
  fold0..4/            每折模型权重
  pred_fold0..4.jsonl  每折在其验证折上的预测
  oof_pred.jsonl       完整 OOF 预测（140,378 行）
  cv_summary.json      交叉验证汇总
  calibration.json     供最终推理使用的校准参数
data/processed/tfidf_full/
  tfidf_clf.joblib     TF-IDF 融合成员
  calibration.json     其校准参数
```

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0+cu132 |
| transformers | 5.20.1 |
| scikit-learn | 1.9.1 |
| GPU | 2 × RTX 5090（32GB，sm_120） |

> 注意：`requirements.txt` 中固定的 `torch` 版本仅适用于上述环境。若目标机器是 Blackwell 架构（50 系），必须使用 CUDA 12.8+ / torch 2.7+ 的构建，**不要**直接 `pip install -r` 覆盖镜像自带的 torch。

## 复现步骤

```bash
# 1. 数据预处理（按 group 生成 train / holdout）
python scripts/preprocess.py

# 2. 5 折全量微调（脚本自带按折跳过，中断后重跑只补未完成的折）
FOLDS=5 BATCH=16 ACCUM=1 MAXLEN=512 EPOCHS=2 bash train_rented.sh

# 脚本会自动执行：
#   merge_cv.py  → oof_pred.jsonl + cv_summary.json
#   calibrate.py → calibration.json

# 3. 生成提交结果
python scripts/predict.py --input data/val.jsonl --out-csv result.csv \
    --ensemble-dirs data/processed/cls_full_5fold/fold0,data/processed/cls_full_5fold/fold1,data/processed/cls_full_5fold/fold2,data/processed/cls_full_5fold/fold3,data/processed/cls_full_5fold/fold4 \
    --calibration data/processed/cls_full_5fold/calibration.json \
    --tfidf-model data/processed/tfidf_full/tfidf_clf.joblib \
    --tfidf-calibration data/processed/tfidf_full/calibration.json \
    --max-len 512 --chunk-long
```

环境冒烟测试（几分钟，验证 GPU 链路）：

```bash
SMOKE=1 bash train_rented.sh
```

`train_rented.sh` 常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODE` | `full` | `full` 全量微调 / `lora` 低显存 |
| `FOLDS` | `5` | `1` 表示不切折，只训单模型 |
| `MAXLEN` | `512` | 训练与推理必须一致 |
| `BATCH` / `ACCUM` | `8` / `2` | 24GB 卡建议 8×2，40GB 卡建议 16×1 |
| `EPOCHS` | `2` | 5 折实测 2 epoch 最优 |
| `SMOKE` | `0` | `1` 只跑 2000 条验证环境 |
| `TRAIN_TFIDF` | `0` | `1` 顺便重训 TF-IDF 融合成员 |

## 合规说明

- **不使用任何形式的验证集/测试集转导式自适应**。`predict.py` 保留 `--prior-match` 参数，但它依赖目标集概率分布，仅用于本地诊断，**提交时禁止使用**。
- 校准参数与先验全部在训练侧 OOF 上拟合，不读取目标集任何统计量。
- 训练与推理的 `max_len` 必须一致，否则截断口径不同会导致概率整体偏移。
- 5 折切分保证同一源文章的所有版本落在同一折，避免同源泄漏抬高交叉验证分数。

## 已知限制

- 同域 AUC 0.999 属饱和区间，无法据此外推比赛分数；跨域（未见领域）性能显著下降，本地 OOD 留出实验中 base 规模模型跨域 AUC 仅约 0.908。
- 先验对齐会改善 F1 口径的判定边界，但会抬高概率校准误差（ECE 0.013 → 0.040）。若评估口径包含概率质量（log loss / Brier），需要重新权衡。
- 长文本分块推理比单次截断慢约 30%，但能挽回尾部信息，长文本子集宏 F1 有可测提升。

## 数据声明

赛题原始数据（`data/*.jsonl`）、模型权重（`*.safetensors`）、训练日志均**不随本仓库分发**，已通过 `.gitignore` 排除。复现需自行按赛题规则获取数据并放置到 `data/` 目录。
