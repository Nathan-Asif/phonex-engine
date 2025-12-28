# Phonex Voice Engine

A FastAPI-based microservice for **Text-to-Speech (TTS)** and **Voice Cloning** powered by MeloTTS and OpenVoice V2.

**Developed by [Nathan Asif](https://github.com/Nathan-Asif)**

---

## Features

- 🗣️ **Text-to-Speech** - High-quality multi-language TTS using MeloTTS
- 🎭 **Voice Cloning** - Clone any voice with OpenVoice V2 tone color conversion
- 🌍 **Multi-language Support** - EN, ES, FR, ZH, JP, KR
- ⚡ **Fast API** - RESTful API with automatic documentation
- 🔄 **Voice Registration** - Register voices once, use them multiple times

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg installed on system

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/phonex-voice-engine.git
cd phonex-voice-engine
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

### 5. Run the Server

```bash
python init.py
```

Or with uvicorn directly:

```bash
uvicorn init:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: `http://localhost:8000`

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
  "speed": 1.0
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

# Register a voice
with open("sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/voice/register",
        data={"voice_name": "my_voice"},
        files={"voice_sample": f}
    )
print(response.json())

# Generate cloned speech
response = requests.post(
    "http://localhost:8000/clone",
    json={"text": "Hello world!", "voice_id": "my_voice"}
)
audio_url = response.json()["audio_url"]

# Download audio
audio = requests.get(f"http://localhost:8000{audio_url}")
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

---

## Project Structure

```
phonex_system/
├── init.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── models/             # AI model files
│   ├── melotts/        # MeloTTS models
│   └── open_voice/     # OpenVoice V2 models
│       └── checkpoints_v2/
├── downloads/          # Generated audio files
├── voices/             # Registered voice embeddings
└── venv/               # Virtual environment
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

---

## Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device ID |

---

## License

MIT License

---

## Credits

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - Text-to-Speech engine
- [OpenVoice](https://github.com/myshell-ai/OpenVoice) - Voice cloning technology
