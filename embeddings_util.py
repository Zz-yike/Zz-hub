"""轻量中文向量嵌入（BGE-small-zh）。首次 ingest 需联网（或配置 HF_ENDPOINT 镜像）；问答时默认仅读本地缓存，避免再次请求 Hub。"""

from langchain_community.embeddings import HuggingFaceEmbeddings

import config


def _resolve_embedding_device() -> str:
    mode = getattr(config, "EMBEDDING_DEVICE", "auto")
    if isinstance(mode, str):
        mode = mode.lower().strip()
    if mode in ("cuda", "gpu"):
        return "cuda"
    if mode == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_embeddings(*, allow_download: bool = True) -> HuggingFaceEmbeddings:
    device = _resolve_embedding_device()
    model_kwargs: dict = {"device": device}
    if not allow_download:
        model_kwargs["local_files_only"] = True
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
