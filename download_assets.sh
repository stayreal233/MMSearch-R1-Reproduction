#!/bin/bash
# ==============================================================================
# MMSearch-R1 权重与数据集一键下载脚本
# 支持 Hugging Face 官方源、hf-mirror 国内镜像源以及 ModelScope 阿里魔塔社区
# ==============================================================================
set -e

BASE_DIR="/root/autodl-tmp"
MODEL_DIR="${BASE_DIR}/models"
DATA_DIR="${BASE_DIR}/datasets"

mkdir -p "${MODEL_DIR}" "${DATA_DIR}"

echo "=========================================================="
echo " 正在准备下载 MMSearch-R1 所需大模型权重与评测数据集..."
echo " 目标存储目录: ${BASE_DIR}"
echo "=========================================================="

# 默认启用国内高速镜像源（如果不需要可注释掉）
export HF_ENDPOINT="https://hf-mirror.com"

# 1. 确保安装 huggingface_hub
pip install -q huggingface_hub datasets

# 2. 下载 MMSearch-R1-7B 主推理模型
echo ""
echo "[1/4] 下载 MMSearch-R1-7B 模型权重..."
huggingface-cli download lmms-lab/MMSearch-R1-7B \
  --local-dir "${MODEL_DIR}/MMSearch-R1-7B" \
  --local-dir-use-symlinks False

# 3. 下载 Qwen2.5-VL-7B-Instruct（基座模型/对照基线）
echo ""
echo "[2/4] 下载 Qwen2.5-VL-7B-Instruct 模型权重..."
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "${MODEL_DIR}/Qwen2.5-VL-7B-Instruct" \
  --local-dir-use-symlinks False

# 4. (可选) 下载 Qwen3-32B-FP8（用于本地实时高质量文本摘要）
echo ""
echo "[3/4] 下载 Qwen3-32B-FP8 摘要模型权重..."
huggingface-cli download Qwen/Qwen3-32B-FP8 \
  --local-dir "${MODEL_DIR}/Qwen3-32B-FP8" \
  --local-dir-use-symlinks False

# 5. 下载 FVQA 评测数据集
echo ""
echo "[4/4] 下载 FVQA 评测数据集..."
huggingface-cli download --repo-type dataset lmms-lab/FVQA \
  --local-dir "${DATA_DIR}/FVQA" \
  --local-dir-use-symlinks False

echo ""
echo "=========================================================="
echo " 所有模型与数据集下载完成！"
echo " 检查结果:"
ls -lh "${MODEL_DIR}"
ls -lh "${DATA_DIR}"
echo "=========================================================="
