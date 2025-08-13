from qdrant_client import QdrantClient
import torch

QDRANT_URL = "https://09a6d049-00c4-4b77-8e95-1dcc9ea5df34.eu-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-ZPZib9FxehqbTuqxsk7QdVjBQd0LlQEq7dpjF1b4PI"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

CORS_SETTINGS = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
