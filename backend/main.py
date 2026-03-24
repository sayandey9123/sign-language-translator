import sys
import os

# Add backend folder to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from utils.inference import predict_sign
from utils.word_builder import WordBuilder
import json
import pickle

app = FastAPI(title="Sign Language Translator API")

# ── BASE DIR ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST ENDPOINTS ──
@app.get("/")
def root():
    return {"message": "Sign Language Translator API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def get_labels():
    with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return {"labels": list(encoder.classes_)}

# ── WEBSOCKET ENDPOINT ──
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")

    word_builder = WordBuilder()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "frame":
                result = predict_sign(message["data"])

                if result["hand_detected"] and result["letter"]:
                    word_state = word_builder.update(
                        result["letter"],
                        result["confidence"]
                    )
                else:
                    word_state = word_builder.get_state()

                await websocket.send_text(json.dumps({
                    "type": "prediction",
                    "letter": result["letter"],
                    "confidence": result["confidence"],
                    "top3": result["top3"],
                    "hand_detected": result["hand_detected"],
                    "word": word_state["current_word"],
                    "sentence": word_state["sentence"],
                    "stable_frames": word_state["stable_frames"],
                    "required_frames": word_state["required_frames"],
                    "letter_committed": word_state["letter_committed"]
                }))

            elif message["type"] == "clear_word":
                if word_builder.current_word:
                 word_builder.sentence += word_builder.current_word + " "
                word_builder.current_word = ""
                await websocket.send_text(json.dumps({
                "type": "word_done",
                "sentence": word_builder.sentence.strip(),
                "word": ""
    }))

    except WebSocketDisconnect:
        print("Client disconnected!")