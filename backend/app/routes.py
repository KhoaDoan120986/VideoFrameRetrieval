from fastapi import APIRouter, Form, UploadFile, File
from typing import Optional
import json, time
from typing import List, Optional
from app.config import DEVICE, CLIP_collection, BGE_collection, GTE_collection
from app.services.search import retrieve_frame, retrieve_from_image, temporal_search, retrieve_have_subtitles, \
    ClipSearch, BGECaptionSearch, GTECaptionSearch, VIDEO_TO_FRAMES

router = APIRouter()

@router.post("/search")
async def api_search(query: Optional[str] = Form(None), topK: int = Form(...), 
                     mode: str = Form("hybrid"), caption_mode: str = Form("bge"), 
                     alpha: float = Form(0.5), file: UploadFile = File(None)):
    """
    Enhanced search API with caption mode support
    - mode: hybrid, clip, vintern, image
    - caption_mode: bge, gte (for vintern and hybrid modes)
    - alpha: text/visual balance for hybrid mode (0.1-0.9)
    """
    try:
        start_time = time.time()
        if mode == "image":
            if file is None:
                return {"error": "No file uploaded for image mode"}
            contents = await file.read()
            results = retrieve_from_image(contents=contents, topK=topK)
            search_info = f"IMAGE search"
        else:
            if query is None or query.strip() == "":
                return {"error": "No query provided for text mode"}
            results = retrieve_have_subtitles(query=query, topK=topK, mode=mode,
                                    caption_mode=caption_mode, alpha=alpha)
            # results = retrieve_frame(query=query, topK=topK, mode=mode,
            #                         caption_mode=caption_mode, alpha=alpha)
            search_info = f"{mode.upper()} mode with {caption_mode.upper()} model"

        duration = time.time() - start_time
        formatted_results = [{"image": r["id"], "caption": f"{r['id']} | Score: {r['score']:.2f}"} for r in results]
        return {
            "results": formatted_results,
            "search_info": {
                "mode": mode,
                "caption_mode": caption_mode if mode in ["hybrid", "vintern"] else None,
                "alpha": alpha if mode == "hybrid" else None,
                "duration": round(duration, 3),
                "count": len(results),
                "description": search_info
            }
        }
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@router.post("/temporal_search")
async def api_temporal_search(events: str = Form(...), topK: int = Form(100), 
                              mode: str = Form("hybrid"), caption_mode: str = Form("bge"), 
                              alpha: float = Form(0.5), search_mode: str = Form("progressive")):
    """
    Enhanced Temporal search API for TRAKE mode
    - events: JSON array of sequential event descriptions
    - search_mode: "progressive" (frontend compatible) or "consolidated" (TRAKE visualization)
    - Returns different result formats based on search_mode
    """
    try:
        events_list = json.loads(events)
        if not isinstance(events_list, list) or len(events_list) == 0:
            return {"error": "Events must be a non-empty list"}

        valid_events = [e.strip() for e in events_list if e.strip()]
        if len(valid_events) == 0:
            return {"error": "No valid events provided"}

        start_time = time.time()
        results = temporal_search(events=valid_events, topK=topK, mode=mode, 
                                  caption_mode=caption_mode, alpha=alpha, search_mode=search_mode)

        duration = time.time() - start_time
        final_count = len(results)
        result_type = "video_timelines"

        return {
            "results": results,
            "search_info": {
                "mode": mode,
                "caption_mode": caption_mode,
                "alpha": alpha if mode == "hybrid" else None,
                "search_mode": search_mode,
                "duration": round(duration, 3),
                "events_processed": len(valid_events),
                "final_count": final_count,
                "result_type": result_type,
                "description": f"{search_mode.title()} temporal search through {len(valid_events)} events"
            }
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for events"}
    except Exception as e:
        return {"error": f"Temporal search failed: {str(e)}"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": {
            "clip": ClipSearch is not None,
            "bge": BGECaptionSearch is not None,
            "gte": GTECaptionSearch is not None,
            "translator": ClipSearch.translator is not None
        },
        "collections": {
            "clip": CLIP_collection,
            "bge": BGE_collection,
            "gte": GTE_collection
        },
        "frame_count": sum(map(len, VIDEO_TO_FRAMES.values())),
        "video_count": len(VIDEO_TO_FRAMES),
        "supported_search_modes": ["progressive", "consolidated"]
    }
