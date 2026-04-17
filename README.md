 RAG Demo：企业制度文档问答

基于 **LangChain + Chroma + BGE 中文向量** 的检索增强生成（RAG）示例：从 **Word（.docx）** 入库，支持 **多轮对话**、**历史感知检索**，并针对 **考核等级 / 分数表** 与 **申诉流程** 做了多路锚点与等级池回退。可选 **FastAPI** 与 **MCP**，便于接入 **Dify** 等外部系统。

## 功能概览

- 递归加载项目内 `.docx`（含段落与表格），切块后写入本地 **Chroma** 向量库  
- 主检索：**MMR**；补充：**等级锚点**、**申诉锚点**、**等级池（Grade Pool）**；支持并行与条件跳过以降延迟  
- 记忆：**ConversationSummaryMemory** + 最近若干轮原文，减轻摘要丢失数字与档位的问题  
- 大模型：兼容 **OpenAI API**（示例为火山 Ark）  
- 可选：**FastAPI**（`POST /chat`）、**FastMCP**（HTTP + `/mcp`，供 Dify MCP 使用）

## 环境要求

- Python **3.10+**（建议 3.11）  
- 首次向量化需能下载嵌入模型（可配置 `HF_ENDPOINT` 镜像）；问答阶段可仅读本地缓存  

## 安装

```bash
cd ragdemo
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置

在项目根目录创建 **`.env`**（勿提交到 Git），示例：

```env
OPENAI_API_KEY=你的API密钥
ARK_ENDPOINT_ID=你的推理端点或模型ID
OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3

# 可选：国内 HuggingFace 镜像
# HF_ENDPOINT=https://hf-mirror.com

# 可选：向量库与路径
# CHROMA_PERSIST_DIR=./chroma_db
# EMBEDDING_DEVICE=auto
```

更多可调参数见 **`config.py`**（检索 k 值、锚点问句、并行线程等）。

## 使用流程

### 1. 入库（ingest）

将 `.docx` 放在项目目录或 `documents/` 下，执行：

```bash
python ingest.py
```

会生成/更新 **`chroma_db/`**（或 `CHROMA_PERSIST_DIR` 指定目录）。更新文档后建议重新执行；需全量重建时可删除该目录再 ingest。

### 2. 命令行对话

```bash
python chat.py
```

空行退出。

### 3. HTTP API（FastAPI）

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

- `POST /chat`：JSON `{"message": "问题", "session_id": "可选，多轮建议固定"}`  
- `GET /health`：健康检查  
- Swagger：`http://127.0.0.1:8000/docs`

### 4. MCP（供 Dify 等，需先启动 API）

PowerShell 示例：

```powershell
$env:RAG_API_BASE = "http://127.0.0.1:8000"
$env:MCP_TRANSPORT = "http"
$env:MCP_HTTP_HOST = "0.0.0.0"
$env:MCP_HTTP_PORT = "8765"
python mcp_server.py
```

Dify 添加 **HTTP MCP** 时，服务器 URL 请使用 **`http://127.0.0.1:8765/mcp`**（勿使用 `0.0.0.0` 作为客户端地址；Docker 内可用 `http://host.docker.internal:8765/mcp`）。

## 项目结构

| 文件 | 说明 |
|------|------|
| `config.py` | 从 `.env` 加载配置与检索/锚点参数 |
| `document_loader.py` | 解析 `.docx`（段落 + 表格） |
| `embeddings_util.py` | BGE-small-zh 嵌入与设备选择 |
| `ingest.py` | 入库脚本 |
| `chat.py` | RAG 链与命令行入口 |
| `api.py` | FastAPI 服务 |
| `mcp_server.py` | MCP 工具（内部调用 `/chat`） |
| `requirements.txt` | Python 依赖 |

## 注意事项

- **不要将 `.env`、API Key、内部制度原文** 提交到公开仓库；`.gitignore` 已忽略 `.env`、`chroma_db/`、`.venv/`。  
- 若仓库中含示例 `.docx`，请注意版权与保密要求。  

## 许可证

自用/学习示例；使用第三方库请遵循各自开源协议。
