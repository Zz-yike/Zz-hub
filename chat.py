"""加载向量库，Stuff 合并 + ConversationSummaryMemory 对话摘要 + 历史感知检索。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from textwrap import dedent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryMemory
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

import config
from embeddings_util import get_embeddings


def _build_history_for_chain(memory: ConversationSummaryMemory) -> str:
    """摘要（易丢数字）+ 最近若干轮原文（保留 89 分、良好等具体结论）。"""
    summary = memory.load_memory_variables({}).get("history") or "（尚无摘要）"
    msgs = memory.chat_memory.messages
    if not msgs:
        return f"【滚动摘要】\n{summary}\n\n【最近对话原文】\n（尚无）"
    tail = msgs[-config.RECENT_DIALOG_MAX_MESSAGES :]
    lines: list[str] = []
    for m in tail:
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"助手：{m.content}")
    recent = "\n".join(lines) if lines else "（尚无）"
    return (
        f"【滚动摘要】\n{summary}\n\n"
        f"【最近对话原文】（具体数字、档位、推理前提以本节为准；与摘要冲突时以本节为准）\n"
        f"{recent}"
    )


def _dedupe_docs_preserve_order(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        fp = f"{d.metadata.get('source', '')}|{(d.page_content or '')[:500]}"
        if fp not in seen:
            seen.add(fp)
            out.append(d)
    return out


_APPEAL_TRIGGER = re.compile(r"申诉|异议|不服|复核|觉得不公|评分不公|遗漏")
# 与「分数/等级/考核场景」相关时才跑等级锚点与等级池，避免纯申诉句白跑一遍
_GRADE_SCORE_TRIGGER = re.compile(
    r"考核|等级|优秀|良好|合格|不合格|分数|评分|总分|满分|扣分|绩效|福利|档|业绩|专员|运维|"
    r"故障|文档|处理|错误|达标|评优|核定|奖金|晋升|补贴|待改进"
)


def _text_has_grade_score_mapping(text: str) -> bool:
    """判断片段是否同时涉及等级词与常见分数，避免误把仅有‘优秀’无分数的句当等级表。"""
    if not text or len(text) < 6:
        return False
    t = re.sub(r"\s+", "", text)
    patterns = [
        r"优秀.{0,100}(90|100|九十分|一百分)",
        r"(90|100).{0,100}优秀",
        r"良好.{0,100}(80|89|八十分)",
        r"(80|89).{0,100}良好",
        r"合格.{0,100}(70|79|七十分)",
        r"(70|79).{0,100}合格",
        r"不合格.{0,60}(70|六十)",
        r"90[-~～至到]\s*100",
        r"80[-~～至到]\s*89",
        r"70[-~～至到]\s*79",
    ]
    return any(re.search(p, t) for p in patterns)


def _inject_grade_pool_docs(
    vectordb: Chroma,
    merged: list[Document],
    *,
    user_q: str = "",
) -> list[Document]:
    if not config.GRADE_POOL_FALLBACK_ENABLED:
        return merged
    if config.GRADE_POOL_CONDITIONAL and user_q and not _GRADE_SCORE_TRIGGER.search(user_q):
        return merged
    blob = "\n".join(d.page_content or "" for d in merged)
    if _text_has_grade_score_mapping(blob):
        return merged
    try:
        pool = vectordb.similarity_search(
            config.GRADE_POOL_QUERY,
            k=config.GRADE_POOL_SEARCH_K,
        )
    except Exception:
        return merged
    hits = [d for d in pool if _text_has_grade_score_mapping(d.page_content or "")]
    hits = hits[: config.GRADE_POOL_MAX_INJECT]
    if not hits:
        return merged
    return _dedupe_docs_preserve_order(hits + merged)


def _should_skip_grade_anchors(user_q: str) -> bool:
    if not config.GRADE_ANCHOR_CONDITIONAL:
        return False
    if not _APPEAL_TRIGGER.search(user_q):
        return False
    if _GRADE_SCORE_TRIGGER.search(user_q):
        return False
    return True


def _parallel_fetch_anchor_queries(retriever, queries: list[str]) -> list[Document]:
    """多路锚点并行检索，缩短 wall-clock（本轮总嵌入次数不变，更易吃满 CPU/IO）。"""
    if not queries:
        return []
    workers = min(
        max(1, config.RETRIEVAL_PARALLEL_WORKERS),
        max(1, len(queries)),
    )

    def one(q: str) -> list[Document]:
        try:
            return list(retriever.invoke(q))
        except Exception:
            return []

    out: list[Document] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, q) for q in queries]
        for fut in as_completed(futs):
            out.extend(fut.result())
    return out


def create_conversation_memory(llm: ChatOpenAI) -> ConversationSummaryMemory:
    """与 CLI / API 共用：每个会话单独一个 Memory 实例。"""
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="history",
        return_messages=False,
        input_key="input",
        output_key="answer",
        human_prefix="用户",
        ai_prefix="助手",
    )


def build_rag_chain_core():
    """构建共享 RAG 链与 LLM（无会话记忆）。API 按 session_id 为每个会话创建 Memory。"""
    if not config.OPENAI_API_KEY or not config.ARK_ENDPOINT_ID:
        raise RuntimeError("请先配置 .env 中的 OPENAI_API_KEY 与 ARK_ENDPOINT_ID")

    embeddings = get_embeddings(allow_download=False)
    vectordb = Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVE_K,
            "fetch_k": config.RETRIEVE_FETCH_K,
            "lambda_mult": config.RETRIEVE_MMR_LAMBDA,
        },
    )

    llm = ChatOpenAI(
        model=config.ARK_ENDPOINT_ID,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
        temperature=0.1,
    )

    contextualize_q_system_prompt = (
        "你是检索查询改写助手。给定对话中的历史消息与用户最新问题，"
        "若问题中有「这、那、上面、之前、优秀」等依赖前文的指代，请改写为一条"
        "可单独用于文献检索的完整中文问句；若问题本身已经自洽，则保持原意稍作规范化即可。\n"
        "特别注意：若用户在反驳或补充**对话里已说的个人情境**（如"
        "「我不是说过只有一处文档错误吗」「我前面就是那个意思」），"
        "检索问句仍应围绕**考核项、扣分、等级线、总分、申诉流程**等政策内容，"
        "不要改写成「制度正文是否重复排版/章节重复」等——除非用户明确在问文件版式问题。"
        "只输出这一条问句，不要解释。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    # 多路 similarity 锚点：等级表 + 用户问申诉时的「五、申诉流程」等（与主问句向量常不对齐）
    _sim_k = max(config.GRADE_ANCHOR_PER_QUERY_K, config.APPEAL_ANCHOR_PER_QUERY_K)
    retriever_sim_anchor = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": _sim_k},
    )

    def _merged_retrieve(inputs: dict) -> list[Document]:
        user_q = str(inputs.get("input", "") or "")
        skip_grade = _should_skip_grade_anchors(user_q)

        def run_primary() -> list[Document]:
            return list(history_aware_retriever.invoke(inputs))

        def run_grade_anchors() -> list[Document]:
            if not config.GRADE_ANCHOR_ENABLED or skip_grade:
                return []
            return _parallel_fetch_anchor_queries(
                retriever_sim_anchor,
                list(config.GRADE_ANCHOR_QUERIES),
            )

        def run_appeal_anchors() -> list[Document]:
            if not config.APPEAL_ANCHOR_ENABLED or not _APPEAL_TRIGGER.search(user_q):
                return []
            return _parallel_fetch_anchor_queries(
                retriever_sim_anchor,
                list(config.APPEAL_ANCHOR_QUERIES),
            )

        max_workers = 3 if (config.GRADE_ANCHOR_ENABLED and not skip_grade) else 2
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_p = ex.submit(run_primary)
            fut_a = ex.submit(run_appeal_anchors)
            fut_g = (
                ex.submit(run_grade_anchors)
                if config.GRADE_ANCHOR_ENABLED and not skip_grade
                else None
            )
            primary = fut_p.result()
            appeal_raw = fut_a.result()
            anchor_raw = fut_g.result() if fut_g is not None else []

        appeal_docs = _dedupe_docs_preserve_order(appeal_raw)[: config.APPEAL_ANCHOR_MERGE_MAX]
        anchor = (
            _dedupe_docs_preserve_order(anchor_raw)[: config.GRADE_ANCHOR_MERGE_MAX]
            if config.GRADE_ANCHOR_ENABLED and not skip_grade
            else []
        )
        merged = _dedupe_docs_preserve_order(appeal_docs + anchor + primary)
        merged = _inject_grade_pool_docs(vectordb, merged, user_q=user_q)
        return merged[: config.MERGED_RETRIEVE_CAP]

    retrieval_merged = RunnableLambda(_merged_retrieve)

    # 系统提示：分层精简，减少重复；{context}{history} 由 Stuff/ Memory 注入
    system_prompt = dedent(
        """
        你是企业制度问答助手。

        【信源】事实与条文以「检索上下文」为准。该上下文**前部**常含补回的「等级/分数段」或「申诉流程/条件」等专题片段，请**通读**后再判断是否「未载明」。下方「对话记忆」= 滚动摘要 + 最近原文；**最近原文**中的具体数字与缺省前提往往比摘要更可靠。除已标明的推演/缺省外，勿将检索未写明的内容说成既定事实。
        【多轮一致】若最近原文中已给出总分、等级或演算前提，本轮必须沿用**同一套前提与结论**（例如前文已认定 89 分、良好，后文不得无说明地改换另一套计分表得到 46 分）。若必须改用算法或假设，须先写明「与此前不同之处在于…」。

        【缺省满分——全会话强制】凡涉及总分、档位、或与档位挂钩的福利：**同一 session 内统一遵守**，不得问着问着就悄悄丢掉。
        - 检索片段里**未写分值且未写扣罚细则**的考核模块 → 演算时**默认按该模块满分**计（并在回答中标明「此为推演假设」即可，不要宣称「文档规定默认满分」）。
        - 用户说过「其它均合格」或延续此前已采用缺省满分的推演 → **后续每一问（含福利）都必须沿用**，**禁止**再以「某模块满分未写清楚」为由**单方面弃用缺省**、改口「档位无法判定」。唯有用户**明确说**要换假设（例如不要默认满分）时才能改，且须写明原因。
        - 福利题：先按**与上文一致的缺省规则**定档位（或承认最近原文已定的档），再对照各档福利；**禁止**用「未规定其它等级福利」绕开前文已得出的档位而不点名沿用关系。

        【口令：其它/缺失＝默认满分】用户一旦说出「其它默认满分」「未写模块默认满分」「缺的按满分」等（含近似说法），从该句起必须执行，**不得**再用「保守估计」架空：
        - 凡条文写「达标得A–B分」类区间 → 本次演算一律取 **B（该项满分/区间上限）**，**禁止**私自改取A或「保守最低分」。
        - 若上下文体现总分为100而已列出分项满分之和为 S（S<100），差额 **(100−S)** 视为**未在片段展开的模块整体按缺省满分**一次性计入，得到**唯一总分**，**禁止**再引入未知数 Y 或「Y∈[0,100−S]」模糊带过。
        - 在此口令下须给出**确定档位**及**该档福利**（写明依赖用户「默认满分」指令）；**禁止**在已收口令后仍以「无法确定福利」为主结论，除非用户当场撤回该前提。

        【评分与等级】若问题涉及能否达到某等级，且上下文中出现可加分的数字：
        - 先写清假设（含是否启用缺省满分），再列算式（数字须出自上下文或缺省规则），再对比等级线。
        - 禁止用「未说明总分/权重」作为全文唯一回复并完全拒绝演算。
        - 仅当上下文完全没有分项分/扣分数字、且也无法在合理缺省下推演时，才可以「无法从当前片段回答」作主结论。

        【等级映射——禁止自相矛盾】
        - 一旦推出**确定总分 T**（如100），**必须**在同一条「检索上下文」全文里查找是否出现**分数区间与等级名称的对照**（如「90-100」「优秀」等出现在同一文档片段中即可）。
        - **若能找到对照**：把 T 套入区间，给出**明确等级名称**（例：T=100 且存在优秀线含100 → **优秀**）。**禁止**在已写明 T 且上下文里明明有区间表的情况下，仍写「无法判定等级」。
        - **若整段检索文字中确无任一「分数区间↔等级」表述**（系统已向量化库扩大检索并筛入可能的等级条后仍无）：可答「推演总分为 T；**当前向量库检索结果中未见等级—分数对照条文**」，并提示核对制度全文是否以图片/扫描件呈现或需重新 ingest；不得空泛说「制度无标准」。**对话记忆**里若已有等级划分，映射时须沿用（多轮一致）。

        【推演补充】已写明满分的项除用户点名扣分外按满分；片段未出现且用户称「其它均合格」的模块可按满分辅助假设并声明以制度全文为准；文中对「合格」若另有固定分值则优先用该分值。

        【福利待遇·条文复述】上下文若载有按等级划分的待遇，须在分析中**完整列出各档**要点；某档未出现在片段中则写「片段未载明该档福利」，勿编造。

        【两类错误】①用户自述角色/失误次数等属**会话前提**，依摘要承接，勿与制度 Word 的版式问题混淆。②仅当用户明确问文件错别字/重复时，才讨论制度文本瑕疵。

        【版式】先分析/引用/算式，最后收束为三行且其后禁止再接正文：第一行「---」；第二行「**【最终结果】**」；第三行起用**加粗**写对用户所问的直接结论 1～2 句（含拒答原因）。
        【禁止】编造上下文没有的分值与福利；勿把假设说成条文。

        对话记忆：
        {history}

        检索上下文：
        {context}
        """
    ).strip()
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(retrieval_merged, document_chain)
    return chain, llm


def build_chain():
    """命令行：单进程单会话。"""
    try:
        chain, llm = build_rag_chain_core()
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    memory = create_conversation_memory(llm)
    return chain, memory


def main() -> None:
    chain, memory = build_chain()
    print(
        "RAG 就绪（摘要记忆 + 最近原文 + 历史感知检索 + 等级/申诉锚点）"
        + "。输入问题，空行退出。\n"
    )
    while True:
        q = input("请输入你要查询的问题: ").strip()
        if not q:
            break
        hist = _build_history_for_chain(memory)
        out = chain.invoke(
            {
                "input": q,
                "chat_history": memory.chat_memory.messages,
                "history": hist,
            }
        )
        ans = out.get("answer", out)
        print("助手:", ans)
        memory.save_context({"input": q}, {"answer": ans})
        print()


if __name__ == "__main__":
    main()
