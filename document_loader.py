from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document as LCDocument

# 索引时跳过这些目录下的 .docx（如向量库、虚拟环境）
_IGNORE_DIR_NAMES = frozenset({".venv", "venv", "chroma_db", "__pycache__", ".git"})


def _should_skip_docx(path: Path) -> bool:
    return any(part in _IGNORE_DIR_NAMES for part in path.parts)


def _iter_body_blocks(doc: DocxDocument):
    """按 Word 正文顺序遍历顶层段落与表格（表格可向量化，需从此处取出，仅 paragraphs 会漏表）。"""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _block_to_text(block) -> str | None:
    if isinstance(block, Paragraph):
        t = block.text.strip()
        return t if t else None
    if isinstance(block, Table):
        rows: list[str] = []
        for row in block.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            rows.append(" | ".join(cells))
        body = "\n".join(rows)
        return f"[表格]\n{body}" if body.strip() else None
    return None


def _docx_to_plain_text(path: Path) -> str:
    docx = Document(str(path))
    chunks: list[str] = []
    for block in _iter_body_blocks(docx):
        t = _block_to_text(block)
        if t:
            chunks.append(t)
    return "\n".join(chunks)


def load_docx_as_langchain_docs(search_roots: list[Path]) -> list[LCDocument]:
    """从目录中收集全部 .docx（递归），转为 LangChain Document。"""
    paths: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.docx"):
            if not _should_skip_docx(p):
                paths.add(p)

    docs: list[LCDocument] = []
    for path in sorted(paths):
        text = _docx_to_plain_text(path)
        if not text:
            continue
        docs.append(
            LCDocument(
                page_content=text,
                metadata={"source": str(path.resolve())},
            )
        )
    return docs
