from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from app.config import CORS_SETTINGS
import uvicorn
from pyngrok import ngrok
import socket
import threading
import time
import argparse


app = FastAPI(
    title="Video Event Retrieval API v2.0",
    description="Enhanced multimodal search with temporal capabilities"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_SETTINGS["allow_origins"],
    allow_credentials=CORS_SETTINGS["allow_credentials"],
    allow_methods=CORS_SETTINGS["allow_methods"],
    allow_headers=CORS_SETTINGS["allow_headers"],
)

app.include_router(router)

PORT = 8000
HOST = "0.0.0.0"
NGROK_AUTH_TOKEN = "31JYCDRSSloOw7lPnlEos7Y8sTv_5PUsnG81esTd4PAMccnDz"

"""Check if a local TCP port is already in use."""
def is_port_in_use(port: int, host="127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


"""Run FastAPI server"""
def run_server():
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="info", reload=False)


"""Start ngrok tunnel and print URLs"""
def start_ngrok(auth_token: str = None):
    if auth_token:
        ngrok.set_auth_token(auth_token)
        print("✅ Ngrok auth token set")
    try:
        # Clean old tunnels
        for t in ngrok.get_tunnels():
            addr = (t.config or {}).get("addr", "")
            if str(PORT) in addr:
                try:
                    ngrok.disconnect(t.public_url)
                except Exception:
                    pass

        # Limit tunnels
        if len(ngrok.get_tunnels()) >= 3:
            ngrok.kill()

        # Create tunnel
        tunnel = ngrok.connect(addr=PORT, proto="http", bind_tls=True)
        public_url = tunnel.public_url

        print("\n" + "=" * 60)
        print("🌐 BACKEND READY!")
        print(f"📡 Public URL: {public_url}")
        print(f"📖 API Docs: {public_url}/docs")
        print(f"🏥 Health Check: {public_url}/health")
        print(f"💻 Local URL: http://localhost:{PORT}")
        print("\n🎯 COPY THE PUBLIC URL TO YOUR FRONTEND!")
        print("=" * 60)

        globals()["PUBLIC_URL"] = public_url
    except Exception as e:
        print(f"❌ Ngrok tunnel failed: {e}")
        print(f"🔧 Server still available locally: http://localhost:{PORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with optional ngrok tunnel")
    parser.add_argument("--ngrok", type=str, default=None, help="Ngrok auth token")
    args = parser.parse_args()
    if args.ngrok:
        NGROK_AUTH_TOKEN = args.ngrok

    if not is_port_in_use(PORT):
        print(f"🚀 Starting FastAPI server on {HOST}:{PORT}")
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)
        print("✅ Server started successfully")
        start_ngrok(NGROK_AUTH_TOKEN)
    else:
        print(f"🔁 Server already running on http://localhost:{PORT}")
