from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_SETTINGS
from app.search.routes import search_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_SETTINGS["allow_origins"],
    allow_credentials=CORS_SETTINGS["allow_credentials"],
    allow_methods=CORS_SETTINGS["allow_methods"],
    allow_headers=CORS_SETTINGS["allow_headers"],
)

app.include_router(search_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
