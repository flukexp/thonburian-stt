# Thonburian STT Server

A server for Thai speech recognition using the [Whisper-TH-Medium-Combined](https://huggingface.co/biodatlab/whisper-th-medium-combined) model from BioDat Lab. This server provides high-quality Thai speech-to-text transcription capabilities through a simple API built with FastAPI. The model supports both Thai and English, with optimization for Thai language audio transcription.

## Features

- **Thai-optimized speech recognition** using the Whisper-TH model
- **Simple API** for transcribing audio files
- **Automatic model downloading** from Hugging Face Hub
- **Supports both CPU and GPU inference**
- **Cross-Origin Resource Sharing (CORS) enabled**
- **Batch processing** for efficient transcription of longer audio files

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/thonburian-stt.git
cd thonburian-stt
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Server

### Start the FastAPI Server

```bash
python server.py
```

The server will start at `http://0.0.0.0:8081`.

## API Endpoints

### Health Check

**Endpoint:** `GET /`

**Response:**
```json
{
  "status": "Thonburian STT Server is running"
}
```

### Transcribe Audio

**Endpoint:** `POST /transcribe`

**Parameters:**
- `file` (file, required): The audio file to transcribe
- `language` (string, optional, default: `th`): The language code (e.g., `th` for Thai)
- `task` (string, optional, default: `transcribe`): The task type

**Example Request:**
```bash
curl -X POST "http://localhost:8081/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.mp3" \
  -F "language=th" \
  -F "task=transcribe"
```

**Example Response:**
```json
{
  "text": "สวัสดีครับ นี่คือการทดสอบระบบรู้จำเสียงภาษาไทย",
  "processing_time": 2.34
}
```

### Alternative STT Endpoint

**Endpoint:** `POST /stt`

**Parameters:**
- `file` (file, required): The audio file to transcribe
- Request body (optional):
  ```json
  {
    "language": "th",
    "task": "transcribe"
  }
  ```

## Configuration

### Running on GPU

If a CUDA-compatible GPU is available, the model will automatically use it for inference. Otherwise, it defaults to CPU.

```python
device = 0 if torch.cuda.is_available() else "cpu"
```

### Model Caching

The required model files are automatically downloaded from Hugging Face and cached in the `model_cache` directory.

## Logging

Logging is enabled by default and provides real-time updates on:
- Model loading status
- Transcription requests
- Processing time
- Errors and exceptions

## Performance

The server uses the Whisper-TH model with the following optimizations:
- Chunk-based processing for handling longer audio files
- Batch processing for efficient inference
- Automatic device selection (GPU/CPU)

## Supported Audio Formats

The server can process various audio formats, including:
- WAV
- MP3
- OGG
- FLAC
- M4A

## Future Improvements

- Add support for streaming audio input
- Implement custom vocabulary and language models
- Add speaker diarization capabilities
- Implement confidence scores for transcriptions