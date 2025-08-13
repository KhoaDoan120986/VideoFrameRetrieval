from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from app.config import qdrant_client, DEVICE
from app.search.AI_models.bge_embedding import BGEEmbedding
from app.search.AI_models.clip_embedding import CLIPEmbedding
from app.search.AI_models.translation import Translator
from collections import defaultdict
import heapq

# model_name = "BAAI/bge-m3"
# model_name = "AITeamVN/Vietnamese_Embedding"
# model_name = "AITeamVN/Vietnamese_Reranker"
bge_embed_model = BGEEmbedding(model_name="BAAI/bge-small-en", device=DEVICE)
bge_vector_store = QdrantVectorStore(client=qdrant_client, collection_name="MSVD")
bge_index = VectorStoreIndex.from_vector_store(vector_store=bge_vector_store, embed_model=bge_embed_model)

clip_embed_model = CLIPEmbedding(device=DEVICE)
clip_vector_store = QdrantVectorStore(client=qdrant_client, collection_name="image")
clip_index = VectorStoreIndex.from_vector_store(vector_store=clip_vector_store, embed_model=clip_embed_model)

translator = Translator(device=DEVICE)
def retrieve_frame(query: str, topK: int, alpha: float = 0.8):
    prompt = translator.translate(query, source_lang="vi")

    bge_nodes = bge_index.as_retriever(similarity_top_k=topK).retrieve(prompt)
    clip_nodes = clip_index.as_retriever(similarity_top_k=topK).retrieve(prompt)
    
    combined_scores = defaultdict(float)
    for node in bge_nodes:
        combined_scores[node.metadata["id"]] += node.score * alpha

    for node in clip_nodes:
        combined_scores[node.metadata["id"]] += node.score * (1 - alpha)

    top_results = heapq.nlargest(topK, combined_scores.items(), key=lambda x: x[1])

    return [
        {"image": video_id.strip(), "caption": f"{video_id} | Score: {score:.2f}"}
        for video_id, score in top_results
    ]
