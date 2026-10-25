﻿﻿﻿﻿﻿﻿# 一键复现脚本（Windows PowerShell）
#
# 目的：把"数据 -> 训练 -> 校准 -> 提交文件"整条链路固化成可重复执行的命令，
#      并统一固定随机种子，满足赛题"模型可复现"的硬性要求。
#
# 用法：
#   .\repro.ps1 -Stage all          # 全流程（训练约 2.2 小时）
#   .\repro.ps1 -Stage preprocess   # 只做数据预处理
#   .\repro.ps1 -Stage calibrate    # 只重算概率校准
#   .\repro.ps1 -Stage predict      # 用官方 val.jsonl 生成 result.csv
#
# 前置：
#   1. 已安装 requirements.txt 中的依赖（GPU 版 torch 需 CUDA 12.8 索引）
#   2. 原始数据位于 data/data.jsonl
#   3. 官方验证集放到 data/val.jsonl（赛前不公开）

param(
    [ValidateSet("all", "preprocess", "train", "calibrate", "predict")]
    [string]$Stage = "all",
    [string]$ValFile = "data/val.jsonl",
    [string]$OutCsv  = "result.csv"
)

$ErrorActionPreference = "Stop"
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内镜像，避免下载骨干模型超时
$env:PYTHONHASHSEED = "42"

$ModelDir = "data/processed/cls_xlmr_large_lora"
$TfidfDir = "data/processed/tfidf_full"
$Seed = 42

function Run-Step($Name, $Block) {
    Write-Host "`n===== $Name =====" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$Name 失败，退出码 $LASTEXITCODE" }
}

if ($Stage -eq "all" -or $Stage -eq "preprocess") {
    Run-Step "1/5 数据预处理（分组切分，防止同源泄漏）" {
        python scripts/preprocess.py --holdout-ratio 0.1 --min-chars 20 --seed $Seed
    }
}

if ($Stage -eq "all" -or $Stage -eq "train") {
    Run-Step "2/5 训练：xlm-roberta-large + LoRA（8GB 显存配置）" {
        python scripts/train_cls.py `
            --model xlm-roberta-large `
            --lora --lora-r 16 --lora-alpha 32 --lora-targets query,value `
            --max-len 256 --batch 8 --grad-accum 4 --epochs 2 --lr 1e-4 `
            --seed $Seed --outdir $ModelDir
    }

    Run-Step "3/5 训练：TF-IDF 融合成员（全量，保存 joblib）" {
        python scripts/baseline_tfidf.py `
            --train data/processed/train.jsonl `
            --holdout data/processed/holdout.jsonl `
            --outdir $TfidfDir --max-train 0 `
            --save-model "$TfidfDir/tfidf_clf.joblib"
    }
}

if ($Stage -eq "all" -or $Stage -eq "calibrate") {
    Run-Step "4/5 概率校准（Platt 缩放，两个成员各自校准）" {
        python scripts/calibrate.py --pred "$ModelDir/pred_holdout.jsonl" --out "$ModelDir/calibration.json"
        python scripts/calibrate.py --pred "$TfidfDir/baseline_pred_holdout.jsonl" --out "$TfidfDir/calibration.json"
    }
}

if ($Stage -eq "all" -or $Stage -eq "predict") {
    if (-not (Test-Path $ValFile)) {
        Write-Host "`n[跳过] 未找到 $ValFile，官方验证集到手后再执行：" -ForegroundColor Yellow
        Write-Host "  python scripts/predict.py --input $ValFile --out-csv $OutCsv --tfidf-model $TfidfDir/tfidf_clf.joblib --tfidf-calibration $TfidfDir/calibration.json" -ForegroundColor Yellow
    } else {
        Run-Step "5/5 生成提交文件 result.csv（神经+TF-IDF 融合，仅用训练侧校准）" {
            # 注意：不要加 --prior-match。它依赖目标集概率分布，属 transductive 使用验证集，
            # 违反参赛指南"禁止任何形式的验证集数据探查"，只在本地诊断时使用。
            python scripts/predict.py `
                --input $ValFile --out-csv $OutCsv `
                --tfidf-model "$TfidfDir/tfidf_clf.joblib" `
                --tfidf-calibration "$TfidfDir/calibration.json" `
                --fuse-weight 0.5
        }
    }
}

Write-Host "`n完成。" -ForegroundColor Green

