import io
from PIL import Image
from typing import List, Optional
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from qdrant_client import models

class SearchEngine:
    def __init__(self, qdrant_client, collection_name, device, model=None, translator=None):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.device = device
        self.translator = translator

        if model is not None:
            self.embed_model = model
        # self.vector_store = QdrantVectorStore(client=self.qdrant_client, 
        #                                       collection_name=self.collection_name)
        # self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, 
        #                                                 embed_model=self.embed_model)

    def retrieve_from_image(self, contents: bytes, topK: int):
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        vector_query = self.embed_model._get_image_embedding(image)

        clip_nodes = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector_query,
            limit=topK,
            with_payload=True
        )

        results = [
            {"id": node.payload.get("id", "").strip(), "score": node.score}
            for node in clip_nodes
        ]
        return results

    def retrieve(self, query: str, topK: int, frame_ids: Optional[List] = None):
        if 'Subtitle' in self.collection_name:
            topK = topK // 2
        query_text = query

        if self.translator:
            query_text = self.translator.translate(query, source_lang="vi")

        vector_query = self.embed_model._get_text_embedding(query_text)
        query_filter = None
        if frame_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="id", match=models.MatchAny(any=frame_ids))]
            )

        nodes = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector_query,
            limit=topK, 
            with_payload=True,
            query_filter=query_filter)
        
        if 'Subtitle' in self.collection_name:
            results = [
                {"id": frame_idx, "score": node.score}
                for node in nodes
                for frame_idx in node.payload.get("frame_list", [])
            ]
        else:
            results = [
                {"id": node.payload["id"].strip(), "score": node.score}
                for node in nodes
            ]   

        return results