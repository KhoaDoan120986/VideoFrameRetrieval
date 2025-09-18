import os
import io
from collections import defaultdict
from typing import List, Optional

import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import DEVICE
from app.clients.qdrant_clients import qdrant_client
from app.services.search_engine import SearchEngine
from .translator import Translator
from .embeddings import Embedding


VIDEO_TO_FRAMES  = defaultdict(list)
offset = None

while True:
    result, offset = qdrant_client.scroll(
        collection_name="Image",
        scroll_filter=None,
        with_payload=True,
        limit=5000,
        offset=offset
    )

    for point in result:
        fid = point.payload.get("id")
        if fid:
            vid = "_".join(fid.split("_")[:2])
            VIDEO_TO_FRAMES [vid].append(fid)

    if offset is None:
        break

translator = Translator(device=DEVICE)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-H-14-quickgelu",
    pretrained="dfn5b",
    device=DEVICE
)
tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
clip_model = clip_model.eval()
clip_embedder = Embedding(clip_model, model_name="ViT-H-14-quickgelu", device=DEVICE,
                          preprocess=clip_preprocess, tokenizer=tokenizer, model_type="clip")
ClipSearch = SearchEngine(qdrant_client=qdrant_client, collection_name="Image")

gg_model = SentenceTransformer("google/embeddinggemma-300m")
caption_embedder = Embedding(gg_model,model_name="google/embeddinggemma-300m",
                             device=DEVICE, model_type="caption")
CaptionSearch = SearchEngine(qdrant_client=qdrant_client, collection_name="Caption")



def retrieve_frame(query: str, topK: int, mode: str = "hybrid",
                   alpha: float = 0.5, frame_ids: Optional[List] = None):
    if mode == "clip":
        clip_query = translator.translate(query)
        clip_vector_query = clip_embedder._get_query_embedding(clip_query)
        clip_nodes = ClipSearch.retrieve_with_vector(
            vector_query=clip_vector_query, 
            topK=topK,
            frame_ids=frame_ids
        )
        return clip_nodes
    
    elif mode == "vintern":
        caption_vector_query = caption_embedder._get_query_embedding(query)
        caption_nodes = CaptionSearch.retrieve_with_vector(
            vector_query=caption_vector_query, 
            topK=topK,
            frame_ids=frame_ids
        )
        return caption_nodes

    else: 
        clip_query = Translator.translate(query)
        clip_vector_query = clip_embedder._get_query_embedding(clip_query)
        clip_nodes = ClipSearch.retrieve_with_vector(
            vector_query=clip_vector_query, 
            topK=topK,
            frame_ids=frame_ids
        )

        caption_vector_query = caption_embedder._get_query_embedding(query)
        caption_nodes = CaptionSearch.retrieve_with_vector(
            vector_query=caption_vector_query, 
            topK=topK,
            frame_ids=frame_ids
        )

        combined_scores = defaultdict(float)
        weights= (alpha, 1 - alpha)
        for nodes, w in ((caption_nodes, weights[0]), (clip_nodes, weights[1])):
            for node in nodes:
                combined_scores[node["id"]] += node["score"] * w

        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:topK]

        return [{"id": video_id, "score": score} for video_id, score in top_results]


def retrieve_from_image(contents: bytes, topK: int):
    """
    Image-based search using CLIP embeddings
    """
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    clip_vector_query = clip_embedder._get_image_embedding(image)
    results = ClipSearch.retrieve_with_vector(
        vector_query=clip_vector_query,
        topK=topK,
        frame_ids=None
    )
    return results

def parse_image_name(image_name: str):
    parts = image_name.split("_", 2)
    vid = f"{parts[0]}_{parts[1]}"
    frame_idx = int(parts[2])
    return vid, frame_idx

def group_by_video(final_results, num_events):
    video_event_data = {}
    for e_idx, results in enumerate(final_results):
        for item in results:
            vid, _ = parse_image_name(item["id"])
            ev_list = video_event_data.setdefault(vid, [[] for _ in range(num_events)])
            ev_list[e_idx].append(item)

    filtered_data = {
        vid: {i: frames for i, frames in enumerate(ev_list)}
        for vid, ev_list in video_event_data.items()
        if all(ev_list)
    }
    return filtered_data

def compute_video_score(video_events, device=DEVICE):
    event_lengths = []
    event_avgs = []

    for frames in video_events.values():
        if not frames:
            continue
        scores = torch.tensor([f["score"] for f in frames], device=device)
        event_avgs.append(torch.mean(scores))
        event_lengths.append(len(frames))

    if not event_avgs:
        return 0.0
        
    lengths_tensor = torch.tensor(event_lengths, dtype=torch.float32, device=device)
    avgs_tensor = torch.stack(event_avgs)

    event_scores = avgs_tensor * torch.log1p(lengths_tensor)
    max_possible = torch.sum(torch.log1p(lengths_tensor))
    video_score = torch.sum(event_scores) / (max_possible + 1e-12)

    return video_score.item()


def beam_search(video_event, beam_size=3, length_norm=True):
    num_events = len(video_event)
    for frames in video_event.values():
        for f in frames:
            if "frame_index" not in f:
                _, idx = parse_image_name(f["id"])
                f["frame_index"] = idx

    sequences = [([f], f["score"]) for f in video_event.get(0, [])]
    if not sequences:
        return []

    for e_idx in range(1, num_events):
        next_frames = video_event.get(e_idx, [])
        if not next_frames:
            break

        candidates = [
            (seq + [f], seq_score + f["score"])
            for seq, seq_score in sequences
            for f in next_frames
            if f["frame_index"] > seq[-1]["frame_index"]
        ]

        if not candidates:
            break
            
        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
    if length_norm and sequences:
        sequences = [(seq, score / len(seq)) for seq, score in sequences]

    return [
        [{"id": f["id"], "score": f["score"]} for f in seq]
        for seq, score in sequences
        if len(seq) == num_events
    ]
    
def temporal_search(events: List[str], topK: int = 100,
                    mode: str = "hybrid", caption_mode: str = "bge",
                    alpha: float = 0.5, search_mode: str = "progressive"):
    final_results = []  

    if search_mode == "progressive":
        frame_ids = None
        for event in events:
            results = retrieve_frame(query=event, topK=topK, mode=mode, 
                                     alpha=alpha, frame_ids=frame_ids)
            final_results.append(results)
            video_ids = {parse_image_name(item['id'])[0] for item in results}
            frame_ids = [f for vid in video_ids for f in VIDEO_TO_FRAMES[vid]]
            
    else: #consolidated
        for event in events:
            results = retrieve_frame(query=event, topK=topK, mode=mode, 
                                     alpha=alpha, frame_ids=None)
            final_results.append(results)

    video_event_data = group_by_video(final_results, len(events))

    results_list = []
    for vid, events_dict in video_event_data.items():
        score = compute_video_score(events_dict, device=DEVICE)
        sequences = beam_search(events_dict, length_norm=True)
    
        if sequences:
            results_list.append({
                "video": vid,
                "score": score,
                "frame_sequence": sequences
            })

    results_list = sorted(results_list, key=lambda x: x["score"], reverse=True)
    return results_list