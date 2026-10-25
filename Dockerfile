# 可复现运行环境（GPU 版）
#
# 构建：
#   docker build -t textsource-detect .
#
# 推理（需要把数据和模型目录挂载进来）：
#   docker run --gpus all -v $PWD/data:/app/data textsource-detect \
#       python3 scripts/predict.py --input /app/data/val.jsonl --out-csv /app/data/result.csv
#
# 说明：
#   - 基镜像固定 CUDA 12.8，与 RTX 5060 (Blackwell) 和 torch 2.11.0+cu128 匹配。
#   - 骨干模型 xlm-roberta-large 默认从 HuggingFace 下载，首次运行需要网络；
#     若需完全离线，可在构建时把模型缓存 COPY 进 /root/.cache/huggingface，
#     或用 --base-model 指向本地目录。
#   - 训练与推理都固定随机种子 seed=42，详见各脚本。

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages \
        -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

COPY scripts ./scripts

# 模型目录（LoRA 适配器 + 分词器 + calibration.json）与数据目录通过挂载提供
ENV MODEL_DIR=/app/data/processed/cls_xlmr_large_lora

CMD ["python3", "scripts/predict.py", "--input", "/app/data/val.jsonl", "--out-csv", "/app/data/result.csv"]
