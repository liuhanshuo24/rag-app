#!/usr/bin/env bash
# RAG 全栈问答助手 - 一键启动 (Linux)
set -e
cd "$(dirname "$0")"

echo "RAG 全栈问答助手 - 一键启动 (Linux)"
echo "----------------------------------"

# 1. 创建虚拟环境
if [ ! -d .venv ]; then
    echo "[1/4] 创建虚拟环境 .venv ..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 安装依赖
echo "[2/4] 安装依赖 ..."
pip install -r requirements.txt -q

# 国内网络下载 HuggingFace 模型较慢，使用镜像加速（首次建索引时生效）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 准备配置
if [ ! -f .env ]; then
    echo "[3/4] 首次运行：请将 .env.example 复制为 .env 并填入 LLM_API_KEY"
    cp .env.example .env
fi

# 4. 构建索引并启动
echo "[4/4] 构建知识库索引 ..."
python scripts/ingest.py

echo
echo "启动服务: http://127.0.0.1:8000"
echo "按 Ctrl+C 停止"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
