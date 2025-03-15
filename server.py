from datetime import datetime
import logging
import os
import torch
import io
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

class STTRequest(BaseModel):
    language: str = "th"
    task: str = "transcribe"
    
class STTResponse(BaseModel):
    text: str
    processing_time: float

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Model definitions
MODEL_NAME = "biodatlab/whisper-th-medium-combined"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global variable for preloaded pipeline
asr_pipeline = None

def load_model_files():
    """Downloads model files from Hugging Face."""
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    for file in MODEL_FILES:
        try:
            hf_hub_download(repo_id=MODEL_NAME, filename=file, cache_dir=cache_dir)
        except Exception as e:
            logging.warning(f"Could not download {file}: {e}")
    
    logging.info(f"Model files downloaded to {cache_dir}")
    return cache_dir

def load_pipeline():
    """Loads the ASR pipeline once at startup for reuse."""
    global asr_pipeline
    if asr_pipeline is not None:
        return  # Already loaded
    
    cache_dir = load_model_files()
    device = 0 if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Loading ASR pipeline on device: {device}")
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
        cache_dir=cache_dir
    )
    logging.info("ASR pipeline loaded successfully.")

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "Thonburian STT Server is running"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "th", task: str = "transcribe"):
    """Transcribes audio file to text."""
    
    if asr_pipeline is None:
        raise HTTPException(status_code=500, detail="ASR pipeline not loaded.")
    
    logging.info(f"Received audio file: {file.filename}")
    start_time = datetime.now()
    
    try:
        # Read audio file from request
        content = await file.read()
        audio_data = io.BytesIO(content)
        
        # Process the audio file
        result = asr_pipeline(
            audio_data, 
            generate_kwargs={"language": f"<|{language}|>", "task": task}, 
            batch_size=16
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Transcription completed in {duration:.2f} seconds.")
        
        return {
            "text": result["text"],
            "processing_time": duration
        }
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {e}")

@app.post("/stt")
async def stt(file: UploadFile = File(...), request: STTRequest = None):
    """Alternative endpoint for STT with JSON request body."""
    if request is None:
        request = STTRequest()
        
    return await transcribe(file, request.language, request.task)

if __name__ == "__main__":
    load_pipeline()  # Load pipeline once
    uvicorn.run(app, host="0.0.0.0", port=8081)