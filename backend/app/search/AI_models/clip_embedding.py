from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from typing import List
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class CLIPEmbedding(BaseEmbedding):
    _model = PrivateAttr()
    _preprocess = PrivateAttr()
    _tokenizer = PrivateAttr()
    _device = PrivateAttr()

    def __init__(self, model_name: str = "hf-hub:apple/DFN2B-CLIP-ViT-B-16", device: str = "cpu"):
        super().__init__()
        self._device = device
        self._model, self._preprocess = create_model_from_pretrained(model_name)
        self._tokenizer = get_tokenizer("ViT-B-16")
        self._model = self._model.to(self._device).eval()

    def _encode_text(self, text: str) -> List[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens) 
        return emb[0].cpu().numpy().tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode_text(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)