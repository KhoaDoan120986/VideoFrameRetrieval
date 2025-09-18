import torch
import open_clip
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
from PIL import Image
from typing import List

class CLIPEmbedding(BaseEmbedding):
    _model = PrivateAttr()
    _preprocess = PrivateAttr()
    _tokenizer = PrivateAttr()
    _device = PrivateAttr()

    def __init__(self, model_name: str = "ViT-H-14-quickgelu", device: str = "cpu"):
        super().__init__()
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="dfn5b",
            device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model = self._model.to(self._device).eval()

    # --- Text embeddings ---
    def _encode_text(self, text: str) -> List[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().numpy().tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode_text(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    # --- Image embeddings ---
    def _encode_image(self, image: Image.Image) -> List[float]:
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().numpy().tolist()

    def _get_image_embedding(self, image: Image.Image) -> List[float]:
            return self._encode_image(image)

    async def _aget_image_embedding(self, image: Image.Image) -> List[float]:
        return self._get_image_embedding(image)

class CaptionEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en", device: str = "cpu", trust_remote_code: bool = False):
        super().__init__()
        self._model = SentenceTransformer(model_name, device=device,
                                          trust_remote_code=trust_remote_code)
        self._model = self._model.eval()


    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)