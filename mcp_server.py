"""MCP 服务：向 Dify 等客户端暴露「制度问答」工具，内部通过 HTTP 调用 FastAPI。

使用前请先启动 API，例如：
  uvicorn api:app --host 0.0.0.0 --port 8000

本进程默认 stdio（适合 Dify / 桌面客户端「本地命令」方式）：
  python mcp_server.py

HTTP 模式（Dify「MCP 服务器 URL」必须用带路径的地址）：
  MCP_TRANSPORT=http MCP_HTTP_PORT=8765 python mcp_server.py
  Dify 填写：http://127.0.0.1:8765/mcp（勿省略 /mcp；勿填 http://0.0.0.0/...）
  Dify 若在 Docker 内：http://host.docker.internal:8765/mcp

环境变量：
  RAG_API_BASE  FastAPI 根地址，默认 http://127.0.0.1:8000。
  Docker 内访问宿主机 API：http://host.docker.internal:8000
"""

from __future__ import annotations

import os

import httpx
from fastmcp import FastMCP

RAG_API_BASE = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000").rstrip("/")

mcp = FastMCP(
    name="ragdemo-regulations",
    instructions=(
        "企业制度 RAG：根据已向量化入库的 Word 制度文档回答问题。"
        "多轮对话请在每次调用中使用相同 session_id。"
    ),
)


@mcp.tool
def ask_company_regulations(question: str, session_id: str = "default") -> str:
    """根据已入库的制度文档回答用户问题。

    Args:
        question: 用户问题（建议中文，与命令行 chat 一致）。
        session_id: 会话 ID；同一用户连续追问请保持不变。
    """
    payload = {
        "message": question.strip(),
        "session_id": (session_id or "default").strip() or "default",
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{RAG_API_BASE}/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return str(data.get("answer", ""))


@mcp.tool
def rag_service_health() -> str:
    """探测 RAG FastAPI 是否存活（GET /health）。"""
    with httpx.Client(timeout=5.0) as client:
        r = client.get(f"{RAG_API_BASE}/health")
        r.raise_for_status()
        return str(r.json())


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_HTTP_PORT", "8765"))
        # FastMCP 默认挂载在 /mcp，Dify 填根 URL 会 503
        print(
            "[Dify MCP] 服务器 URL 请填: "
            f"http://127.0.0.1:{port}/mcp"
            "（Docker 内用 host.docker.internal 替代 127.0.0.1）"
        )
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()
