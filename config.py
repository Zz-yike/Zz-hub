import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# 国内访问 HuggingFace 超时可在 .env 中设置：HF_ENDPOINT=https://hf-mirror.com
_hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ARK_ENDPOINT_ID = os.getenv("ARK_ENDPOINT_ID", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")

# 轻量中文嵌入（约 33M 参数，CPU 可跑）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 嵌入设备：auto（有 CUDA 则用 GPU）| cpu | cuda —— GPU 可明显加快多路检索时的编码
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()

# 小粒度分块（适合制度条文类文档）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# 评分表易被切成小块：用 MMR 从更多候选里挑 k 条，减轻「只有扣分没有满分表」
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
RETRIEVE_FETCH_K = int(os.getenv("RETRIEVE_FETCH_K", "32"))
# 越大越侧重相关性，越小越多样化；0.5~0.7 常用
RETRIEVE_MMR_LAMBDA = float(os.getenv("RETRIEVE_MMR_LAMBDA", "0.65"))

# 随 prompt 附带的最近对话条数（Human/AI 各算一条），弥补纯摘要丢失具体分数的问题
RECENT_DIALOG_MAX_MESSAGES = int(os.getenv("RECENT_DIALOG_MAX_MESSAGES", "10"))

# 主检索易漏「等级划分」时，用多条固定问句 + similarity 补召回，排在上下文前部
GRADE_ANCHOR_ENABLED = os.getenv("GRADE_ANCHOR_ENABLED", "1").lower() not in ("0", "false", "no")
# 每条锚点问句各自检索的条数（similarity）
GRADE_ANCHOR_PER_QUERY_K = int(os.getenv("GRADE_ANCHOR_PER_QUERY_K", "5"))
# 多路锚点去重后，最多带入几条（再与主检索合并）
GRADE_ANCHOR_MERGE_MAX = int(os.getenv("GRADE_ANCHOR_MERGE_MAX", "10"))
MERGED_RETRIEVE_CAP = int(os.getenv("MERGED_RETRIEVE_CAP", "16"))
# 主检索与多路锚点并行时的线程数（I/O + 库内计算；可略大于锚点条数）
RETRIEVAL_PARALLEL_WORKERS = int(os.getenv("RETRIEVAL_PARALLEL_WORKERS", "8"))
# 纯「申诉」类问句（无等级/分数语义）跳过等级锚点与等级池，省嵌入与带宽且不影响该场景召回
GRADE_ANCHOR_CONDITIONAL = os.getenv("GRADE_ANCHOR_CONDITIONAL", "1").lower() not in ("0", "false", "no")
GRADE_POOL_CONDITIONAL = os.getenv("GRADE_POOL_CONDITIONAL", "1").lower() not in ("0", "false", "no")

_RAW_MULTI = os.getenv("GRADE_ANCHOR_QUERIES", "").strip()
if _RAW_MULTI:
    GRADE_ANCHOR_QUERIES = [s.strip() for s in _RAW_MULTI.split("||") if s.strip()]
else:
    _single = os.getenv("GRADE_ANCHOR_QUERY", "").strip()
    if _single:
        GRADE_ANCHOR_QUERIES = [_single]
    else:
        GRADE_ANCHOR_QUERIES = [
            "员工考核 等级 优秀 良好 合格 不合格 分数 90 100 80 89 70 79 划分标准",
            "优秀 90 100 良好 80 89 合格 70 79 不合格 70 综合考核 等级线",
            "考核结果 绩效等级 分数段 晋升 奖金 优秀评级",
        ]

# 合并后的正文仍无「等级+分数」字样时，扩大 similarity 再按正则筛入片段（向量库里常有但此前未命中）
GRADE_POOL_FALLBACK_ENABLED = os.getenv("GRADE_POOL_FALLBACK_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)
GRADE_POOL_SEARCH_K = int(os.getenv("GRADE_POOL_SEARCH_K", "48"))
GRADE_POOL_MAX_INJECT = int(os.getenv("GRADE_POOL_MAX_INJECT", "6"))
GRADE_POOL_QUERY = os.getenv(
    "GRADE_POOL_QUERY",
    "绩效考核 考核等级 优秀 良好 合格 不合格 90 100 80 89 70 79 分数段 综合得分",
)

# 用户问「申诉」等时多路 similarity 补召回「五、申诉流程」类段落（主问句向量常对不齐章节标题）
APPEAL_ANCHOR_ENABLED = os.getenv("APPEAL_ANCHOR_ENABLED", "1").lower() not in ("0", "false", "no")
APPEAL_ANCHOR_PER_QUERY_K = int(os.getenv("APPEAL_ANCHOR_PER_QUERY_K", "5"))
APPEAL_ANCHOR_MERGE_MAX = int(os.getenv("APPEAL_ANCHOR_MERGE_MAX", "6"))
_RAW_APPEAL = os.getenv("APPEAL_ANCHOR_QUERIES", "").strip()
if _RAW_APPEAL:
    APPEAL_ANCHOR_QUERIES = [s.strip() for s in _RAW_APPEAL.split("||") if s.strip()]
else:
    APPEAL_ANCHOR_QUERIES = [
        "申诉 人力资源部 考核结果 异议 书面申诉 公示 工作日",
        "申诉流程 申诉条件 申诉提交 申诉审核 结果反馈",
        "考核结果公示 申诉 直属领导 重新核算 5.1 5.2",
    ]
