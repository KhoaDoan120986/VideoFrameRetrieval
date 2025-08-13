from fastapi import APIRouter
from app.search.models.search_request import SearchRequest
from app.search.services.search_service import retrieve_frame

router = APIRouter()

@router.post("/search")
def api_search(req: SearchRequest):
    results = retrieve_frame(req.query, req.topK)
    return {"results": results}