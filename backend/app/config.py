import torch
from fastapi.middleware.cors import CORSMiddleware

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# CORS
CORS_SETTINGS = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Collection names
CLIP_collection = "Image"
BGE_collection = {
    'Caption': "BGE_Caption",
    'Subtitle': "BGE_subtitles"
}

GTE_collection = {
    'Caption': "GTE_Caption",
    'Subtitle': "GTE_subtitles"
}
