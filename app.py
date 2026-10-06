import os
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vieneu import Vieneu


app = FastAPI(
    title="Ninh VieNeu API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model một lần khi server khởi động
print("Loading VieNeu-TTS...")

tts = Vieneu(
    mode="v3turbo",
    device="cpu",
)

print("VieNeu-TTS ready!")


class TTSRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Ninh VieNeu API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/tts")
def generate_tts(request: TTSRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is empty"
        )

    if len(text) > 3000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 3000 characters."
        )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    output_path = tmp.name
    tmp.close()

    try:
        # VieNeu SDK
        tts.infer(
            text=text,
            output_file=output_path,
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="speech.wav"
        )

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
