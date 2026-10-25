#!/usr/bin/env bash
# ============================================================
# 租卡训练一键脚本（Linux / AutoDL 等）
#
# 前置：
#   1) 当前目录是项目根目录（能看到 scripts/train_gpu.py）
#   2) 已安装 requirements.txt，且 torch 为 CUDA 12.8 版本
#   3) 已生成 data/processed/train.jsonl 与 holdout.jsonl
#
# 用法：
#   bash train_rented.sh                # 默认：24GB 卡，5 折全量微调
#   SMOKE=1 bash train_rented.sh        # 只跑 2000 条冒烟，验证环境（几分钟）
#   FOLDS=1 bash train_rented.sh        # 不切折，训一个全量模型
#   MODE=lora BATCH=16 bash train_rented.sh
#   TRAIN_TFIDF=1 bash train_rented.sh   # 机器上没有 TF-IDF 模型时顺便重训
#
# 24GB 建议：MODE=full MAXLEN=512 BATCH=8 ACCUM=2
# 40GB 建议：MODE=full MAXLEN=512 BATCH=16 ACCUM=1
# ============================================================
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

MODE="${MODE:-full}"
FOLDS="${FOLDS:-5}"
OUT="${OUT:-data/processed/cls_full_5fold}"
MAXLEN="${MAXLEN:-512}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-2}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-0}"          # 0 = 按模式自动：full 2e-5 / lora 1e-4
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [ ! -f scripts/train_gpu.py ]; then
  echo "请在项目根目录运行本脚本（找不到 scripts/train_gpu.py）"
  exit 1
fi
if [ ! -f data/processed/train.jsonl ]; then
  echo "缺少 data/processed/train.jsonl，请先运行：python scripts/preprocess.py"
  exit 1
fi

echo "===== 环境检查 ====="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, "| 显存 %.1f GB" % (p.total_memory / 1e9))
else:
    raise SystemExit("CUDA 不可用，检查驱动与 torch 版本")
PY

CFG="--mode $MODE --max-len $MAXLEN --batch $BATCH --grad-accum $ACCUM \
     --epochs $EPOCHS --lr $LR --seed $SEED --num-workers $NUM_WORKERS"

if [ "$SMOKE" = "1" ]; then
  echo "===== 冒烟测试（2000 条，1 epoch）====="
  python -u scripts/train_gpu.py --mode "$MODE" --max-train 2000 --max-len 256 \
      --batch 4 --grad-accum 1 --epochs 1 --lr "$LR" --seed "$SEED" \
      --outdir data/processed/smoke_gpu
  echo "冒烟通过。正式训练请去掉 SMOKE=1。"
  exit 0
fi

if [ "$FOLDS" -lt 2 ]; then
  echo "===== 单模型全量训练（holdout 监控）====="
  python -u scripts/train_gpu.py $CFG --outdir "$OUT"
  echo "模型与预测：$OUT"
  exit 0
fi

echo "===== K 折训练：$FOLDS 折，输出到 $OUT ====="
mkdir -p "$OUT"
echo "MODE=$MODE FOLDS=$FOLDS MAXLEN=$MAXLEN BATCH=$BATCH ACCUM=$ACCUM EPOCHS=$EPOCHS LR=$LR SEED=$SEED" \
    > "$OUT/run_config.txt"

for i in $(seq 0 $((FOLDS - 1))); do
  if [ -f "$OUT/metrics_fold$i.json" ]; then
    echo "fold $i 已完成，跳过（删除该文件可重跑）"
    continue
  fi
  echo ""
  echo "----- fold $i / $((FOLDS - 1)) -----"
  python -u scripts/train_gpu.py $CFG \
      --cv-folds "$FOLDS" --cv-index "$i" \
      --seed "$((SEED + i))" \
      --outdir "$OUT"
done

echo ""
echo "===== 合并 OOF 并汇总 ====="
python -u scripts/merge_cv.py --cvdir "$OUT" --folds "$FOLDS" \
    --out "$OUT/oof_pred.jsonl"

echo ""
echo "===== 用 OOF 拟合融合校准参数 ====="
python -u scripts/calibrate.py --pred "$OUT/oof_pred.jsonl" \
    --out "$OUT/calibration.json"

# TF-IDF 融合成员体积只有 17MB，通常直接随项目上传即可；
# 如果租的机器上没有，设置 TRAIN_TFIDF=1 重新训练（CPU，约 8 分钟）。
if [ "${TRAIN_TFIDF:-0}" = "1" ]; then
  echo ""
  echo "===== 重新训练 TF-IDF 融合成员 ====="
  python -u scripts/baseline_tfidf.py \
      --train data/processed/train.jsonl --holdout data/processed/holdout.jsonl \
      --outdir data/processed/tfidf_full --max-train 0 \
      --save-model data/processed/tfidf_full/tfidf_clf.joblib
  python -u scripts/calibrate.py \
      --pred data/processed/tfidf_full/baseline_pred_holdout.jsonl \
      --out data/processed/tfidf_full/calibration.json
fi

DIRS=""
for i in $(seq 0 $((FOLDS - 1))); do
  DIRS="$DIRS$OUT/fold$i,"
done
DIRS="${DIRS%,}"

echo ""
echo "完成。产物："
echo "  $OUT/fold*/            每折模型"
echo "  $OUT/oof_pred.jsonl    干净交叉验证预测"
echo "  $OUT/calibration.json  供最终集成推理使用"
echo ""
echo "最终提交（5 折集成 + TF-IDF 融合，仅训练侧校准）："
echo "  python scripts/predict.py --input data/val.jsonl --out-csv result.csv \\"
echo "      --ensemble-dirs $DIRS \\"
echo "      --calibration $OUT/calibration.json \\"
echo "      --tfidf-model data/processed/tfidf_full/tfidf_clf.joblib \\"
echo "      --tfidf-calibration data/processed/tfidf_full/calibration.json \\"
echo "      --max-len $MAXLEN --chunk-long"
echo ""
echo "跨骨干集成（large + base）改用 logit 加权，权重与目录顺序一一对应："
echo "  --ensemble-mode logit --ensemble-weights 0.7,0.3"
echo ""
echo "注意：不要加 --prior-match。它依赖目标集概率分布，属 transductive 使用验证集，"
echo "      违反参赛指南第130行，仅可用于本地诊断。"
