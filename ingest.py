"""将 documents/ 与项目根目录下的 Word 文件写入 Chroma 向量库。"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from document_loader import load_docx_as_langchain_docs
from embeddings_util import get_embeddings


def main() -> None:
    project_root = Path(__file__).resolve().parent
    (project_root / "documents").mkdir(parents=True, exist_ok=True)
    # 从项目根递归收集 .docx（含根目录下「AI Agent相关岗位考核制度…」及 documents/ 等子目录）
    raw_docs = load_docx_as_langchain_docs([project_root])
    if not raw_docs:
        print("未找到任何 .docx 文件。请将 Word 放在本项目目录下（如根目录或 documents/）。")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    splits = splitter.split_documents(raw_docs)

    embeddings = get_embeddings()

    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"已向量化 {len(splits)} 个文本块，库路径: {config.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
