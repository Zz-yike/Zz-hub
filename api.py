"""FastAPI 封装：供 Dify（HTTP 工具 / MCP 桥）及其他客户端调用。"""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat import (
    _build_history_for_chain,
    build_rag_chain_core,
    create_conversation_memory,
)
from langchain.memory import ConversationSummaryMemory


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(
        default="default",
        max_length=256,
        description="多轮对话请固定同一 ID（如用户 ID + 会话号）",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    chain, llm = build_rag_chain_core()
    app.state.chain = chain
    app.state.llm = llm
    app.state.sessions: dict[str, ConversationSummaryMemory] = {}
    app.state.session_lock = Lock()
    yield


app = FastAPI(title="RAG Demo API", version="0.1.0", lifespan=lifespan)


def _get_memory(session_id: str) -> ConversationSummaryMemory:
    sid = (session_id or "default").strip() or "default"
    with app.state.session_lock:
        if sid not in app.state.sessions:
            app.state.sessions[sid] = create_conversation_memory(app.state.llm)
        return app.state.sessions[sid]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    memory = _get_memory(req.session_id)
    hist = _build_history_for_chain(memory)
    try:
        out = app.state.chain.invoke(
            {
                "input": req.message.strip(),
                "chat_history": memory.chat_memory.messages,
                "history": hist,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    ans = out.get("answer", out)
    ans = ans if isinstance(ans, str) else str(ans)
    memory.save_context({"input": req.message.strip()}, {"answer": ans})
    return ChatResponse(answer=ans, session_id=req.session_id.strip() or "default")


@app.get("/health")
def health():
    return {"status": "ok"}
