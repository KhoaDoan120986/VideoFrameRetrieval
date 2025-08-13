from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from typing import List

class BGEEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en", device: str = "cpu"):
        super().__init__()
        self._model = SentenceTransformer(model_name, device=device)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)