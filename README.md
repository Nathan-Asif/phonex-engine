# Phonex Voice Engine

A FastAPI-based microservice for **Text-to-Speech (TTS)** and **Voice Cloning** powered by MeloTTS and OpenVoice V2.

**Developed by [Nathan Asif](https://github.com/Nathan-Asif)**

📦 **Repository**: [github.com/Nathan-Asif/phonex-engine](https://github.com/Nathan-Asif/phonex-engine.git)

---

## Features

- 🗣️ **Text-to-Speech** - High-quality multi-language TTS using MeloTTS
- 🎭 **Voice Cloning** - Clone any voice with OpenVoice V2 tone color conversion
- 🌍 **Multi-language Support** - EN (US, BR, AU, India), ES, FR, ZH, JP, KR
- ⚡ **Fast API** - RESTful API with automatic documentation
- 🔄 **Voice Registration** - Register voices once, use them multiple times
- ☁️ **Modal GPU Support** - Serverless GPU acceleration via Modal
- 📦 **Long-form Audio** - Background processing for large text (10k+ chars)
- 🎵 **Pitch Control** - Adjust pitch from -12 to +12 semitones

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Phonex Voice Engine                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Client    │───▶│  FastAPI Server  │───▶│  Modal Serverless│   │
│  │  (Frontend) │◀───│    (init.py)     │◀───│    GPU Worker    │   │
│  └─────────────┘    └──────────────────┘    └──────────────────┘   │
│                              │                        │             │
│                              ▼                        ▼             │
│                    ┌─────────────────┐     ┌─────────────────────┐  │
│                    │ Voice Register  │     │     MeloTTS +       │  │
│                    │  (OpenVoice)    │     │   OpenVoice V2      │  │
│                    │   (CPU only)    │     │    (GPU T4)         │  │
│                    └─────────────────┘     └─────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Modes of Operation:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Local CPU** | All processing on local machine | Development, low-traffic |
| **Modal GPU** | Offload TTS/Cloning to Modal serverless | Production, high-performance |

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for local) or CPU
- FFmpeg installed on system
- Modal account (for serverless GPU mode)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nathan-Asif/phonex-engine.git
cd phonex-engine
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Models

Create the models directory structure:

```bash
mkdir -p models/melotts
mkdir -p models/open_voice/checkpoints_v2
```

Download required models:
- **MeloTTS**: Clone from [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- **OpenVoice V2**: Download checkpoints from [OpenVoice GitHub](https://github.com/myshell-ai/OpenVoice)

Place them in the `models/` directory.

### 5. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Modal Authentication (required for Modal mode)
MODAL_TOKEN_ID=YOUR_MODAL_TOKEN_ID
MODAL_TOKEN_SECRET=YOUR_MODAL_TOKEN_SECRET

# Feature Toggles
USE_MODAL=True   # Set to False for local-only mode

# Application Settings
HOST=0.0.0.0
PORT=8000
```

### 6. Run the Server

```bash
python init.py
```

Or with uvicorn directly:

```bash
uvicorn init:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: `http://localhost:8000`

---

## Modal Setup (Serverless GPU)

Modal provides on-demand GPU compute for fast inference without maintaining expensive hardware.

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

This will create `.modal.toml` with your credentials.

### 3. Sync Models to Modal Volume

Upload your local models to Modal's persistent storage:

```bash
modal run modal_worker.py::sync
```

This uploads the `models/` folder to a Modal volume called `phonex-models`.

### 4. Deploy the Worker

Deploy the GPU worker class to Modal:

```bash
modal deploy modal_worker.py
```

This creates the `phonex-worker` app with:
- **GPU**: NVIDIA T4
- **Auto-scaling**: 0-5 containers
- **Cold start**: ~60s (model loading)
- **Warm invoke**: <2s

### 5. Enable Modal in Your Server

Set `USE_MODAL=True` in your `.env` file:

```env
USE_MODAL=True
```

When enabled:
- Voice registration runs locally (CPU)
- TTS and voice cloning are offloaded to Modal GPUs
- Jobs are processed asynchronously with status polling

### Modal Worker Architecture

```python
# modal_worker.py - Key Components

@app.cls(gpu="T4", ...)
class VoiceEngine:
    @modal.enter()
    def init_models(self):
        # Loads MeloTTS + OpenVoice once per container
    
    @modal.method()
    def generate_tts(text, language, speaker_id, speed):
        # Standard TTS → MP3
    
    @modal.method()
    def clone_voice(text, language, base_speaker, target_se_bytes, speed, pitch):
        # Voice cloning with chunking for long text → MP3
```

---

## API Documentation

Interactive docs available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Basic health check |
| GET | `/health` | Detailed health status |

### Text-to-Speech

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tts` | Generate speech from text |
| GET | `/tts/speakers` | List all available speakers |
| GET | `/tts/speakers/{language}` | Get speakers for a language |

**Request Body:**
```json
{
  "text": "Hello, world!",
  "language": "EN",
  "speaker": "EN-US",
  "speed": 1.0
}
```

### Voice Cloning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/voice/register` | Register a new voice |
| GET | `/voice/list` | List registered voices |
| DELETE | `/voice/{voice_id}` | Delete a voice |
| POST | `/clone` | Generate speech with cloned voice |
| POST | `/clone/quick` | One-shot clone (no registration) |

**Register Voice (form-data):**
- `voice_name`: Unique name for the voice
- `voice_sample`: Audio file (WAV/MP3, 10-30 seconds)

**Clone Request Body:**
```json
{
  "text": "Hello with my cloned voice!",
  "voice_id": "my_voice",
  "language": "EN",
  "speaker": "EN-US",
  "speed": 1.0,
  "pitch": 0
}
```

### Job Status (Async Operations)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/job/{job_id}` | Check job status |
| GET | `/status/{audio_id}` | Alias for job status |

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": 100,
  "audio_url": "/audio/uuid",
  "audio_id": "uuid"
}
```

### Audio Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/audio/{audio_id}` | Download generated audio |
| DELETE | `/audio/{audio_id}` | Delete generated audio |

---

## Supported Languages & Speakers

| Language | Code | Available Speakers |
|----------|------|-------------------|
| English | EN | EN-US, EN-BR, EN-AU, EN-INDIA, EN-DEFAULT, EN_NEWEST |
| Spanish | ES | ES |
| French | FR | FR |
| Chinese | ZH | ZH |
| Japanese | JP | JP |
| Korean | KR | KR |

---

## Usage Examples

### Register a Voice (cURL)

```bash
curl -X POST http://localhost:8000/voice/register \
  -F "voice_name=john_doe" \
  -F "voice_sample=@/path/to/sample.wav"
```

### Generate Cloned Speech (cURL)

```bash
curl -X POST http://localhost:8000/clone \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice_id": "john_doe"}'
```

### Python Example

```python
import requests
import time

# Register a voice
with open("sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/voice/register",
        data={"voice_name": "my_voice"},
        files={"voice_sample": f}
    )
print(response.json())

# Generate cloned speech (async with Modal)
response = requests.post(
    "http://localhost:8000/clone",
    json={
        "text": "Hello world! This is a longer text that will be processed.",
        "voice_id": "my_voice",
        "language": "EN",
        "pitch": 2  # Slightly higher pitch
    }
)
result = response.json()
job_id = result["job_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/job/{job_id}").json()
    print(f"Status: {status['status']} - Progress: {status['progress']}%")
    
    if status["status"] == "completed":
        audio_url = status["audio_url"]
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    
    time.sleep(2)

# Download audio
audio = requests.get(f"http://localhost:8000{audio_url}")
with open("output.mp3", "wb") as f:
    f.write(audio.content)
```

---

## Project Structure

```
phonex_system/
├── init.py              # Main FastAPI application
├── modal_worker.py      # Modal serverless GPU worker
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .modal.toml          # Modal authentication config
├── README.md            # This file
├── phonex.service       # Systemd service file
├── models/              # AI model files
│   ├── melotts/         # MeloTTS models
│   └── open_voice/      # OpenVoice V2 models
│       └── checkpoints_v2/
│           ├── converter/
│           └── base_speakers/ses/
├── downloads/           # Generated audio files
├── voices/              # Registered voice embeddings (.pth)
└── venv/                # Virtual environment
```

---

## Production Deployment

### Using Gunicorn (Linux)

```bash
pip install gunicorn
gunicorn init:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Using systemd Service

Create `/etc/systemd/system/phonex.service`:

```ini
[Unit]
Description=Phonex Voice Engine
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/phonex_system
Environment="PATH=/path/to/phonex_system/venv/bin"
ExecStart=/path/to/phonex_system/venv/bin/uvicorn init:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable phonex
sudo systemctl start phonex
```

### With Modal (Recommended for Production)

1. Deploy Modal worker: `modal deploy modal_worker.py`
2. Set `USE_MODAL=True` in `.env`
3. Run FastAPI server on VPS (handles requests, offloads compute to Modal)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MODAL` | False | Enable Modal serverless GPU |
| `MODAL_TOKEN_ID` | - | Modal authentication token ID |
| `MODAL_TOKEN_SECRET` | - | Modal authentication secret |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device ID (local mode) |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `FFmpeg not found` | Install FFmpeg: `apt install ffmpeg` or `brew install ffmpeg` |
| `CUDA not available` | Check PyTorch CUDA installation or use CPU mode |
| `Modal cold start slow` | First request after idle takes ~60s for model loading |
| `Voice registration fails` | Ensure audio is 10-30s of clear speech, WAV/MP3 format |

### Modal Debugging

```bash
# Check Modal app status
modal app list

# View logs
modal app logs phonex-worker

# Test worker locally
modal run modal_worker.py::main
```

---

## License

MIT License

---

## Credits

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Text-to-Speech engine
- [OpenVoice](https://github.com/myshell-ai/OpenVoice) - Voice cloning technology
- [Modal](https://modal.com) - Serverless GPU infrastructure
