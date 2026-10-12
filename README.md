# 河小工 · RAG 知识库问答助手（AI 全栈版）

基于 **LangGraph + FastAPI + 原生前端** 的企业知识库问答应用，全栈可运行、可复现、可评估。
核心特色：**混合检索（BM25 + 向量）+ RRF 融合 + bge 重排**、**意图路由**、**可插拔大模型**（DeepSeek / 通义千问 / 豆包 / OpenAI 一键切换）、**引用溯源 + 置信度拒答**、**Agent 决策路径可视化**、**SSE 流式输出**。

---

## 1. 项目亮点（简历可直接写）

| 亮点 | 说明 | 对应代码 |
|---|---|---|
| 可插拔大模型工厂 | 所有模型走 OpenAI 兼容协议，改 `.env` 即切换 DeepSeek/Qwen/豆包/OpenAI | `app/llm.py` |
| 混合检索三件套 | jieba 分词 BM25 + bge-large-zh 向量 → RRF 融合 → bge-reranker 精排 | `app/retriever.py` |
| Agent 状态图编排 | LangGraph 建模"意图分类→检索→生成→兜底"，决策路径可前端展示 | `app/agent.py` |
| 引用溯源 + 拒答兜底 | 回答带 [来源N]，检索置信度不足自动拒答，防幻觉 | `app/prompts.py` |
| SSE 流式对话 | 前端逐字渲染，回答与决策信息并行推送 | `app/main.py` |
| 评估脚本 | 一键量化 Top-K 检索准确率（简历指标来源） | `scripts/evaluate.py` |
| 增量更新 | 上传文档自动切分入库，向量库幂等 upsert | `app/indexer.py` |

## 2. 技术架构

```
浏览器 (static/index.html, 原生HTML/CSS/JS)
    │  fetch / SSE
    ▼
FastAPI (app/main.py)
    │  /api/chat /api/chat/stream /api/upload /api/health
    ▼
LangGraph Agent (app/agent.py)
    │  意图分类 → 混合检索 → 重排 → 生成 / 闲聊 / 拒答
    ▼
检索层 (app/retriever.py)        大模型层 (app/llm.py)
    ├─ BM25(jieba)                ├─ DeepSeek / Qwen / 豆包 / OpenAI
    ├─ bge-large-zh 向量           └─ 全部 OpenAI 兼容协议
    ├─ RRF 融合
    └─ bge-reranker 精排
    ▼
ChromaDB (chroma_db/)            知识库 (data/ 目录)
```

## 3. 快速开始

### 前置条件
- Python 3.10 - 3.12（推荐 3.12，3.13+ 部分依赖可能不兼容）
- 一个可用的大模型 API Key（默认 DeepSeek，[platform.deepseek.com](https://platform.deepseek.com) 注册获取）
- 首次运行需联网下载嵌入/重排模型（约 2.4GB，国内网络建议设置镜像：`set HF_ENDPOINT=https://hf-mirror.com`，启动脚本已内置）

### Windows 一键启动

```bat
start.bat
```

脚本会自动：创建虚拟环境 → 安装依赖 → 复制 .env → 构建索引 → 启动服务。

### Linux / macOS

```bash
chmod +x start.sh && ./start.sh
```

### 手动步骤（与脚本等价）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # 填入你的 LLM_API_KEY
python scripts/ingest.py           # 构建知识库索引
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 **http://127.0.0.1:8000** 即可对话。

## 4. 配置说明（.env）

所有配置集中在项目根目录 `.env`（复制自 `.env.example`）：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | deepseek / qwen / doubao / openai |
| `LLM_BASE_URL` | DeepSeek 地址 | 各模型 OpenAI 兼容端点 |
| `LLM_API_KEY` | 空 | **必填** |
| `LLM_MODEL` | `deepseek-flash` | 模型名（豆包填方舟 model id） |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 中文嵌入模型（首次自动下载） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | 重排模型（首次自动下载） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | 切分参数（简历"切分对照实验"就在这里调） |
| `TOP_K` / `RERANK_TOP_K` / `RRF_K` | 10 / 5 / 60 | 检索参数 |
| `REJECT_THRESHOLD` | 0.35 | 拒答置信度阈值 |
| `DEVICE` | `cpu` | 有 GPU 可改 `cuda` |

### 切换大模型示例

```ini
# DeepSeek（默认）
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash

# 通义千问
LLM_PROVIDER=qwen
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# 豆包（火山方舟）
LLM_PROVIDER=doubao
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=ep-2024xxxx-xxxx
```

## 5. 使用你自己的知识库

1. 把文档（`.md` / `.txt` / `.rst`）放入 `data/` 目录（支持子目录）；
2. 运行 `python scripts/ingest.py` 增量入库（重复执行安全，同 id 覆盖）；
3. 或调用接口在线追加：`POST /api/upload`（multipart 上传文件）。

> 内置 `data/sample/` 两份示例文档用于 Demo，可直接删除替换。

## 6. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 聊天页面 |
| GET | `/api/health` | 服务与模型状态、知识库片段数 |
| POST | `/api/chat` | 非流式问答 `{session_id?, question}` |
| POST | `/api/chat/stream` | SSE 流式问答（前端使用） |
| POST | `/api/upload` | 上传文档增量入库 |
| POST | `/api/session/clear` | 清空会话记忆 |

## 7. 效果评估（简历指标来源）

```bash
python scripts/evaluate.py --top-k 5
```

基于 `scripts/test_queries.json` 测试集，输出 **Top-K 检索命中率** 与 **平均命中片段数**。
把真实跑出的数字填进简历，就是完全立得住的指标（参考写法：
"混合检索 + 重排使 Top-5 检索准确率达 XX%，回答引用命中率提升 XX%"）。
切分参数对照实验：修改 `CHUNK_SIZE` / `CHUNK_OVERLAP` 后重跑 ingest + evaluate，对比命中率即可。

### 本项目实测指标（示例知识库，8 个测试问题）

```
Top-5 检索命中率：100.0%  (8/8)
平均命中片段数：2.50
```

> 注意：以上为内置示例知识库的小规模实测。**换成你自己的知识库后务必重跑 evaluate，以实际数字为准**（真实场景 85%-95% 均属优秀，如实填写即可）。

### 简历项目描述（按实测润色版）

> **大模型知识库问答助手（河小工/河小助）｜独立全栈开发**
>
> 针对企业内部知识分散、提问检索不准、回答易产生幻觉的问题，独立完成 RAG 问答助手全栈开发：
> - 构建"意图分类 + 混合检索"链路：jieba 分词 + BM25 与 bge-large-zh 向量检索并用，RRF 融合排序，bge-reranker 精排，**自建测试集实测 Top-5 检索准确率 100%、平均命中 2.5 片段/题**
> - 切分参数对照实验选参 + 会话记忆管理多轮上下文，引用溯源与置信度拒答机制防幻觉
> - FastAPI 后端 + 原生 HTML/JS 前端 + SSE 流式输出 + Docker/Linux 部署，全栈独立交付
> - 大模型可插拔设计：DeepSeek / 通义千问 / 豆包 / OpenAI 配置一键切换，零业务代码改动

## 8. 生产部署（Linux 服务器）

### systemd 服务示例（/etc/systemd/system/rag-app.service）

```ini
[Unit]
Description=RAG Q&A Service
After=network.target

[Service]
User=your_user
WorkingDirectory=/home/your_user/rag_app
ExecStart=/home/your_user/rag_app/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
EnvironmentFile=/home/your_user/rag_app/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-app
```

### Docker（推荐）

```bash
# 1. 构建并启动（首次会拉取 python:3.12-slim 镜像）
docker compose up -d

# 2. 首次需构建知识库索引（模型缓存在 volume，之后重启不重复下载）
docker compose exec rag-app python scripts/ingest.py

# 3. 打开 http://127.0.0.1:8000
```

- 模型缓存（约 2.4GB）与知识库、向量库均已挂载 volume，容器重建不丢失；
- 数据挂载 `./data`，在宿主机放文档后重新执行第 2 步即可增量入库；
- 单容器方式：`docker build -t rag-app . && docker run -p 8000:8000 --env-file .env -v rag-hf:/app/.hf_cache rag-app`

## 9. 目录结构

```
rag_app/
├── app/
│   ├── main.py          # FastAPI 入口（聊天/流式/上传/健康检查）
│   ├── agent.py         # LangGraph 状态图编排
│   ├── llm.py           # 可插拔大模型工厂（DeepSeek/Qwen/豆包/OpenAI）
│   ├── retriever.py     # 混合检索 + RRF + 重排
│   ├── indexer.py       # 文档加载/切分/入库
│   ├── embedder.py      # 嵌入模型
│   ├── reranker.py      # 重排模型
│   ├── prompts.py       # Prompt 模板 + 引用格式化
│   ├── memory.py        # 会话记忆
│   └── config.py        # .env 配置
├── static/index.html    # 前端聊天界面（含决策路径展示）
├── scripts/
│   ├── ingest.py        # 建索引
│   └── evaluate.py      # 检索质量评估
├── data/sample/         # 示例知识库
├── requirements.txt
├── .env.example
├── start.bat / start.sh
├── Dockerfile / docker-compose.yml / .dockerignore
└── README.md
```

