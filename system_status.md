# Phonex System Overview

Current architecture of the Phonex Voice Engine microservice.

---

## 🛠 Execution Workflow

The system operates on an **ID-based Request/Download pattern**. While currently synchronous, it mimics a job queue by assigning unique identifiers to every synthesis task.

### 1. The Generation Flow (Step-by-Step)
For both standard TTS and Voice Cloning:
1. **Validation**: API validates language (EN, ES, etc.) and speaker ID.
2. **ID Assignment**: A unique `audio_id` (UUID) is generated immediately.
3. **Engine Selection**:
    - **TTS**: Routes to MeloTTS engine.
    - **Cloning**: Routes to MeloTTS (for base audio) → ToneColorConverter (for style transfer).
4. **Synthesis**:
    - Generates `.wav` into the `downloads/` directory.
    - Filename matches the `audio_id`.
5. **Response**: Returns a JSON object containing the `audio_id` and a relative `audio_url`.

### 2. The Retrieval Flow
1. **Web Request**: The frontend/proxy receives the `audio_id`.
2. **Download**: Client requests `/audio/{audio_id}`.
3. **Cleanup**: System provides endpoints to manually delete audio files from disk once consumed.

---

## 🔬 Core Components

### 1. Backend Service (`init.py`)
- **Framework**: FastAPI
- **Engines**: 
    - **MeloTTS**: Multi-language synthesis.
    - **OpenVoice V2**: tone color embedding extraction and conversion.
- **Workflow Logic**:
    - **Registration**: Audio Sample → `se_extractor` → Saved `.pth` embedding in `voices/`.
    - **Cloning**: Base Speaker Embedding + Target Embedding = Cloned Audio.

### 2. DevOps & Pipeline
- **Continuous Deployment**: GitHub Actions triggers on `main` push.
- **Service Resilience**: `systemd` handles auto-restart on failure.
- **File Management**: Uses local disk as a buffer for synthesis results.

### 3. Frontend Link (`phonex_sandbox`)
- Uses a **PHP Proxy** (`proxy.php`) to bypass CORS and securely route requests from the web interface to the backend.
- **App Logic**: Handles the polling/fetching of the generated ID to display the audio player.

---
*Status as of Dec 28, 2025*

