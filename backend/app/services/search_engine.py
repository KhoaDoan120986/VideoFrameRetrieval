from typing import List, Optional
from qdrant_client import models

class SearchEngine:
    def __init__(self, qdrant_client, collection_name):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def retrieve_with_vector(self, vector_query, topK: int, frame_ids: Optional[List] = None):
        if "Subtitle" in self.collection_name:
            topK = topK // 2

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
            query_filter=query_filter,
        )
        
        if "Subtitle" in self.collection_name:
            results = [
                {"id": str(frame_idx), "score": node.score}
                for node in nodes
                for frame_idx in node.payload.get("frame_list", [])
            ]
        else:
            results = [
                {"id": node.payload.get("id", "").strip(), "score": node.score}
                for node in nodes
            ]   
        return results