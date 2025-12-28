"""
Phonex Voice Engine
===================
FastAPI-based microservice for Text-to-Speech and Voice Cloning.

Integrates:
- MeloTTS for high-quality TTS
- OpenVoice V2 for voice cloning (tone color conversion)
"""

import os
import sys
import uuid
import logging
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Modal and GPU config
USE_MODAL = os.getenv("USE_MODAL", "False").lower() == "true"
if USE_MODAL:
    import modal


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phonex")

# Path configuration
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
DOWNLOADS_DIR = BASE_DIR / "downloads"
VOICES_DIR = BASE_DIR / "voices"  # Store registered voice embeddings
MELOTTS_DIR = MODELS_DIR / "melotts"
OPENVOICE_DIR = MODELS_DIR / "open_voice"
CHECKPOINTS_V2_DIR = OPENVOICE_DIR / "checkpoints_v2"

# Add to path
sys.path.insert(0, str(MELOTTS_DIR))
sys.path.insert(0, str(OPENVOICE_DIR))

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)

# Global model instances
tts_models = {}  # Cache TTS models by language
tone_color_converter = None
device = None

# Speaker to base embedding mapping
SPEAKER_TO_SE = {
    "EN-US": "en-us",
    "EN-BR": "en-br",
    "EN-AU": "en-au",
    "EN-INDIA": "en-india",
    "EN-DEFAULT": "en-default",
    "EN_NEWEST": "en-newest",
    "ES": "es",
    "FR": "fr",
    "ZH": "zh",
    "JP": "jp",
    "KR": "kr",
}
 
# Global job tracking
jobs = {}  # job_id -> {"status": str, "progress": int, "audio_url": str, "error": str}



# ============= Pydantic Models =============

class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str
    language: str = "EN"  # EN, ES, FR, ZH, JP, KR
    speaker: str = "EN-US"  # Speaker ID varies by language
    speed: float = 1.0


class TTSResponse(BaseModel):
    """Response model for TTS requests."""
    success: bool
    audio_url: Optional[str] = None
    audio_id: Optional[str] = None
    message: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning TTS."""
    text: str
    voice_id: str  # Registered voice ID
    language: str = "EN"
    speaker: str = "EN-US"  # Base speaker for TTS
    speed: float = 1.0
    pitch: int = 0  # Pitch shift in semitones (-12 to +12)


class VoiceCloneResponse(BaseModel):
    """Response model for voice cloning."""
    success: bool
    audio_url: Optional[str] = None
    audio_id: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = "completed"
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status checks."""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: int = 0
    audio_url: Optional[str] = None
    audio_id: Optional[str] = None
    error: Optional[str] = None
    estimated_time: Optional[str] = None


class VoiceRegisterResponse(BaseModel):
    """Response model for voice registration."""
    success: bool
    voice_id: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    tts_ready: bool
    voice_clone_ready: bool
    registered_voices: int
    version: str = "2.0.0"


# ============= Helper Functions =============

def get_tts_model(language: str):
    """Get or create TTS model for a language."""
    global tts_models, device
    
    lang_upper = language.upper()
    if lang_upper not in tts_models:
        from melo.api import TTS
        tts_models[lang_upper] = TTS(language=lang_upper, device=device)
        logger.info(f"Loaded TTS model for language: {lang_upper}")
    
    return tts_models[lang_upper]


def get_base_speaker_se(speaker: str):
    """Load base speaker embedding for tone color conversion."""
    import torch
    
    speaker_key = SPEAKER_TO_SE.get(speaker.upper())
    if speaker_key is None:
        # Try direct mapping
        speaker_key = speaker.lower().replace("_", "-")
    
    se_path = CHECKPOINTS_V2_DIR / "base_speakers" / "ses" / f"{speaker_key}.pth"
    if not se_path.exists():
        raise ValueError(f"No base speaker embedding found for: {speaker}")
    
    return torch.load(se_path, map_location=device)


def get_registered_voices():
    """Get list of registered voice IDs."""
    voices = []
    for f in VOICES_DIR.glob("*.pth"):
        voices.append(f.stem)
    return voices


def split_text_into_chunks(text: str, max_chars: int = 1000):
    """Split text into manageable chunks by sentences."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    import re
    # Split by common sentence endings followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If a single sentence is longer than max_chars, split it by length
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i+max_chars].strip())
                current_chunk = ""
            else:
                current_chunk = sentence
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c]


def process_long_form_clone(job_id: str, request: VoiceCloneRequest):
    """Background task to handle chunked generation and stitching."""
    global jobs, tone_color_converter, device
    
    try:
        jobs[job_id]["status"] = "processing"
        
        # 1. Split text
        chunks = split_text_into_chunks(request.text)
        total_chunks = len(chunks)
        jobs[job_id]["total_chunks"] = total_chunks
        
        import torch
        from pydub import AudioSegment
        
        # Load target speaker embedding
        voice_se_path = VOICES_DIR / f"{request.voice_id}.pth"
        target_se = torch.load(voice_se_path, map_location=device)
        source_se = get_base_speaker_se(request.speaker)
        
        chunk_files = []
        
        for i, chunk_text in enumerate(chunks):
            # Update progress
            jobs[job_id]["progress"] = int((i / total_chunks) * 100)
            logger.info(f"Job {job_id}: Processing chunk {i+1}/{total_chunks}")
            
            chunk_id = f"{job_id}_chunk_{i}"
            temp_tts_path = DOWNLOADS_DIR / f"temp_tts_{chunk_id}.wav"
            temp_clone_path = DOWNLOADS_DIR / f"temp_clone_{chunk_id}.wav"
            
            # Generate base TTS
            tts_model = get_tts_model(request.language)
            speaker_ids = tts_model.hps.data.spk2id
            
            logger.info(f"Job {job_id}: Generating TTS for chunk {i+1}")
            tts_model.tts_to_file(
                text=chunk_text,
                speaker_id=speaker_ids[request.speaker],
                output_path=str(temp_tts_path),
                speed=request.speed
            )
            
            # Convert tone color
            logger.info(f"Job {job_id}: Converting tone color for chunk {i+1}")
            tone_color_converter.convert(
                audio_src_path=str(temp_tts_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(temp_clone_path),
                message="Phonex"
            )
            
            chunk_files.append(temp_clone_path)
            
            # Clean up base TTS temp file
            if temp_tts_path.exists():
                temp_tts_path.unlink()
        
        # 2. Stitch chunks
        logger.info(f"Job {job_id}: Stitching {len(chunk_files)} segments...")
        combined = AudioSegment.empty()
        for f in chunk_files:
            segment = AudioSegment.from_wav(str(f))
            combined += segment
            # Clean up chunk file after loading
            if f.exists():
                f.unlink()
        
        # 3. Save final output as MP3 (optimized for web delivery)
        output_path = DOWNLOADS_DIR / f"{job_id}.mp3"
        combined.export(str(output_path), format="mp3", bitrate="128k")
        
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "audio_id": job_id,
            "audio_url": f"/audio/{job_id}"
        })
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })
        # Attempt cleanup
        if 'chunk_files' in locals():
            for f in chunk_files:
                if f.exists(): f.unlink()


# ============= Lifecycle =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup, cleanup on shutdown."""
    global tts_models, tone_color_converter, device
    
    logger.info("🚀 Starting Phonex Voice Engine v2.0...")
    
    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if USE_MODAL:
        logger.info("☁️ Using Modal Serverless GPU for inference.")
        logger.info("📝 Loading OpenVoice locally for voice registration only...")
        device = "cpu"  # VPS orchestrator stays on CPU
    else:
        logger.info(f"Using device: {device}")
        # Initialize MeloTTS (only if not using Modal)
        try:
            from melo.api import TTS
            tts_models["EN"] = TTS(language="EN", device=device)
            logger.info("✅ MeloTTS (EN) initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MeloTTS: {e}")
    # Initialize OpenVoice V2 ToneColorConverter
    try:
        from openvoice.api import ToneColorConverter
        
        converter_config = CHECKPOINTS_V2_DIR / "converter" / "config.json"
        converter_ckpt = CHECKPOINTS_V2_DIR / "converter" / "checkpoint.pth"
        
        if converter_config.exists() and converter_ckpt.exists():
            tone_color_converter = ToneColorConverter(
                str(converter_config), 
                device=device
            )
            tone_color_converter.load_ckpt(str(converter_ckpt))
            logger.info("✅ OpenVoice V2 ToneColorConverter initialized successfully")
        else:
            logger.warning("⚠️ OpenVoice V2 checkpoints not found")
            tone_color_converter = None
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenVoice: {e}")
        tone_color_converter = None
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Phonex Voice Engine...")
    tts_models.clear()
    tone_color_converter = None


# ============= FastAPI App =============

app = FastAPI(
    title="Phonex Voice Engine",
    description="Text-to-Speech and Voice Cloning API powered by MeloTTS and OpenVoice V2",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Endpoints =============

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="online",
        tts_ready=len(tts_models) > 0,
        voice_clone_ready=tone_color_converter is not None,
        registered_voices=len(get_registered_voices())
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy" if len(tts_models) > 0 else "degraded",
        tts_ready=len(tts_models) > 0,
        voice_clone_ready=tone_color_converter is not None,
        registered_voices=len(get_registered_voices())
    )


async def run_modal_tts(audio_id: str, request: TTSRequest):
    """Background task to run TTS on Modal."""
    import asyncio
    
    def _sync_modal_call():
        """Synchronous Modal call to run in thread."""
        VoiceEngine = modal.Cls.from_name("phonex-worker", "VoiceEngine")
        engine = VoiceEngine()
        return engine.generate_tts.remote(
            text=request.text,
            language=request.language,
            speaker_id=request.speaker,
            speed=request.speed
        )
    
    try:
        jobs[audio_id]["status"] = "processing"
        logger.info(f"Job {audio_id}: Starting Modal TTS inference...")
        
        # Run Modal call in thread pool to avoid blocking event loop
        wav_bytes = await asyncio.to_thread(_sync_modal_call)
        
        # Save output
        output_path = DOWNLOADS_DIR / f"{audio_id}.mp3"
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
            
        jobs[audio_id].update({
            "status": "completed",
            "progress": 100,
            "audio_url": f"/audio/{audio_id}"
        })
        logger.info(f"Job {audio_id}: Modal TTS completed")
        
    except Exception as e:
        logger.error(f"Job {audio_id} failed on Modal: {e}")
        jobs[audio_id].update({
            "status": "failed",
            "error": str(e)
        })


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Convert text to speech (Asynchronous via Modal if enabled).
    """
    audio_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[audio_id] = {
        "status": "queued",
        "progress": 0,
        "audio_id": audio_id,
        "audio_url": None,
        "error": None
    }

    if USE_MODAL:
        background_tasks.add_task(run_modal_tts, audio_id, request)
        return TTSResponse(
            success=True,
            audio_url=f"/audio/{audio_id}",
            audio_id=audio_id,
            message="Job queued for Modal GPU inference"
        )
    
    # LOCAL SYNC FALLBACK
    try:
        tts_model = get_tts_model(request.language)
        output_path = DOWNLOADS_DIR / f"{audio_id}.mp3"
        speaker_ids = tts_model.hps.data.spk2id
        
        if request.speaker not in speaker_ids:
            raise HTTPException(status_code=400, detail=f"Invalid speaker '{request.speaker}'")
        
        tts_model.tts_to_file(
            text=request.text,
            speaker_id=speaker_ids[request.speaker],
            output_path=str(output_path),
            speed=request.speed
        )
        
        jobs[audio_id]["status"] = "completed"
        jobs[audio_id]["audio_url"] = f"/audio/{audio_id}"
        
        return TTSResponse(
            success=True,
            audio_url=f"/audio/{audio_id}",
            audio_id=audio_id,
            message="Audio generated successfully (Local CPU)"
        )
    except Exception as e:
        jobs[audio_id]["status"] = "failed"
        jobs[audio_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/speakers")
async def get_speakers():
    """Get available TTS speakers/voices for all loaded languages."""
    try:
        speakers = {}
        for lang, model in tts_models.items():
            speaker_ids = model.hps.data.spk2id
            speakers[lang] = list(speaker_ids.keys())
        return {"speakers": speakers, "loaded_languages": list(tts_models.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/speakers/{language}")
async def get_speakers_for_language(language: str):
    """Get available speakers for a specific language."""
    try:
        tts_model = get_tts_model(language)
        speaker_ids = tts_model.hps.data.spk2id
        return {
            "speakers": list(speaker_ids.keys()),
            "language": language.upper()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Voice Cloning Endpoints =============

@app.post("/voice/register", response_model=VoiceRegisterResponse)
async def register_voice(
    voice_name: str = Form(..., description="Unique name for this voice"),
    voice_sample: UploadFile = File(..., description="Audio file (WAV/MP3) of the voice to clone")
):
    """
    Register a new voice for cloning.
    
    Upload an audio sample (WAV, MP3) to create a reusable voice profile.
    The voice sample should be 10-30 seconds of clear speech.
    """
    if tone_color_converter is None:
        raise HTTPException(
            status_code=503, 
            detail="Voice cloning not available. OpenVoice V2 not initialized."
        )
    
    try:
        from openvoice import se_extractor
        
        # Sanitize voice name
        voice_id = voice_name.lower().replace(" ", "_").replace("-", "_")
        
        # Check if voice already exists
        voice_se_path = VOICES_DIR / f"{voice_id}.pth"
        if voice_se_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_id}' already exists. Use a different name or delete the existing one."
            )
        
        # Save uploaded file temporarily
        temp_audio_path = DOWNLOADS_DIR / f"temp_{uuid.uuid4()}{Path(voice_sample.filename).suffix}"
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(voice_sample.file, f)
        
        try:
            # Extract speaker embedding from voice sample
            target_se, audio_name = se_extractor.get_se(
                str(temp_audio_path), 
                tone_color_converter, 
                vad=True
            )
            
            # Save the speaker embedding
            import torch
            torch.save(target_se, voice_se_path)
            
            logger.info(f"Registered voice: {voice_id}")
            
            return VoiceRegisterResponse(
                success=True,
                voice_id=voice_id,
                message=f"Voice '{voice_id}' registered successfully"
            )
        finally:
            # Clean up temp file
            if temp_audio_path.exists():
                temp_audio_path.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/list")
async def list_voices():
    """List all registered voices available for cloning."""
    voices = get_registered_voices()
    return {
        "voices": voices,
        "count": len(voices)
    }


@app.delete("/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a registered voice."""
    voice_path = VOICES_DIR / f"{voice_id}.pth"
    
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    try:
        voice_path.unlink()
        return {"success": True, "message": f"Voice '{voice_id}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_modal_clone(audio_id: str, request: VoiceCloneRequest):
    """Background task to run Voice Cloning on Modal."""
    import asyncio
    
    try:
        jobs[audio_id]["status"] = "processing"
        
        # Load embedding bytes from local storage
        voice_se_path = VOICES_DIR / f"{request.voice_id}.pth"
        if not voice_se_path.exists():
            raise FileNotFoundError(f"Voice {request.voice_id} not found")
        
        with open(voice_se_path, "rb") as f:
            target_se_bytes = f.read()
        
        def _sync_modal_call():
            """Synchronous Modal call to run in thread."""
            VoiceEngine = modal.Cls.from_name("phonex-worker", "VoiceEngine")
            engine = VoiceEngine()
            return engine.clone_voice.remote(
                text=request.text,
                language=request.language,
                base_speaker=request.speaker,
                target_se_bytes=target_se_bytes,
                speed=request.speed,
                pitch=request.pitch
            )
        
        # Run Modal call in thread pool to avoid blocking event loop
        logger.info(f"Job {audio_id}: Starting Modal Clone inference...")
        wav_bytes = await asyncio.to_thread(_sync_modal_call)
        
        # Save output
        output_path = DOWNLOADS_DIR / f"{audio_id}.mp3"
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
            
        jobs[audio_id].update({
            "status": "completed",
            "progress": 100,
            "audio_url": f"/audio/{audio_id}"
        })
        logger.info(f"Job {audio_id}: Modal Clone completed")
        
    except Exception as e:
        logger.error(f"Job {audio_id} failed on Modal: {e}")
        jobs[audio_id].update({
            "status": "failed",
            "error": str(e)
        })


@app.post("/clone", response_model=VoiceCloneResponse)
async def clone_voice_tts(request: VoiceCloneRequest, background_tasks: BackgroundTasks):
    """
    Generate speech with a cloned voice (Async via Modal if enabled).
    """
    audio_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[audio_id] = {
        "status": "queued",
        "progress": 0,
        "audio_id": audio_id,
        "audio_url": None,
        "error": None
    }

    if USE_MODAL:
        background_tasks.add_task(run_modal_clone, audio_id, request)
        return VoiceCloneResponse(
            success=True,
            job_id=audio_id,
            status="queued",
            message="Job queued for Modal GPU inference"
        )
    
    # LOCAL ASYNC FALLBACK (Existing logic for long-form)
    if tone_color_converter is None:
        raise HTTPException(status_code=503, detail="Local Voice cloning not initialized")
        
    # Check if voice exists
    voice_se_path = VOICES_DIR / f"{request.voice_id}.pth"
    if not voice_se_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found")
        
    background_tasks.add_task(process_long_form_clone, audio_id, request)
    return VoiceCloneResponse(
        success=True,
        job_id=audio_id,
        status="queued",
        message="Job queued for local CPU inference"
    )

@app.get("/status/{audio_id}", response_model=JobStatusResponse)
async def get_status_alias(audio_id: str):
    """Alias for get_job_status as requested."""
    return await get_job_status(audio_id)


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of a long-form synthesis job."""
    if job_id not in jobs:
        # Fallback: Check if file already exists (handles server restarts)
        audio_path = DOWNLOADS_DIR / f"{job_id}.mp3"
        if not audio_path.exists():
            audio_path = DOWNLOADS_DIR / f"{job_id}.wav"
            
        if audio_path.exists():
            return JobStatusResponse(
                job_id=job_id,
                status="completed",
                progress=100,
                audio_id=job_id,
                audio_url=f"/audio/{job_id}"
            )
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        audio_id=job.get("audio_id"),
        audio_url=job.get("audio_url"),
        error=job.get("error")
    )


@app.post("/clone/quick", response_model=VoiceCloneResponse)
async def clone_voice_quick(
    text: str = Form(..., description="Text to synthesize"),
    voice_sample: UploadFile = File(..., description="Audio sample of voice to clone"),
    language: str = Form("EN", description="Language: EN, ES, FR, ZH, JP, KR"),
    speaker: str = Form("EN-US", description="Base speaker ID"),
    speed: float = Form(1.0, description="Speech speed")
):
    """
    One-shot voice cloning without registration.
    
    Upload a voice sample and text to generate cloned speech in a single request.
    Useful for quick tests, but slower than using registered voices.
    """
    if tone_color_converter is None:
        raise HTTPException(
            status_code=503, 
            detail="Voice cloning not available. OpenVoice V2 not initialized."
        )
    
    try:
        import torch
        from openvoice import se_extractor
        
        # Generate unique IDs
        audio_id = str(uuid.uuid4())
        temp_audio_path = DOWNLOADS_DIR / f"temp_voice_{audio_id}{Path(voice_sample.filename).suffix}"
        temp_tts_path = DOWNLOADS_DIR / f"temp_tts_{audio_id}.wav"
        output_path = DOWNLOADS_DIR / f"{audio_id}.wav"
        
        # Save uploaded voice sample
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(voice_sample.file, f)
        
        try:
            # Extract target speaker embedding
            target_se, _ = se_extractor.get_se(
                str(temp_audio_path), 
                tone_color_converter, 
                vad=True
            )
            
            # Generate base TTS audio
            tts_model = get_tts_model(language)
            speaker_ids = tts_model.hps.data.spk2id
            
            if speaker not in speaker_ids:
                available = list(speaker_ids.keys())
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid speaker '{speaker}'. Available: {available}"
                )
            
            tts_model.tts_to_file(
                text=text,
                speaker_id=speaker_ids[speaker],
                output_path=str(temp_tts_path),
                speed=speed
            )
            
            # Load source speaker embedding
            source_se = get_base_speaker_se(speaker)
            
            # Convert tone color
            tone_color_converter.convert(
                audio_src_path=str(temp_tts_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path),
                message="Phonex"
            )
            
            logger.info(f"Generated quick clone audio: {audio_id}")
            
            return VoiceCloneResponse(
                success=True,
                audio_url=f"/audio/{audio_id}",
                audio_id=audio_id,
                message="Audio generated successfully"
            )
            
        finally:
            # Clean up temp files
            for p in [temp_audio_path, temp_tts_path]:
                if p.exists():
                    p.unlink()
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Audio Endpoints =============

@app.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Download generated audio file."""
    # Support both formats, prioritizing MP3
    audio_path = DOWNLOADS_DIR / f"{audio_id}.mp3"
    if not audio_path.exists():
        audio_path = DOWNLOADS_DIR / f"{audio_id}.wav"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    mime_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"
    
    return FileResponse(
        path=str(audio_path),
        media_type=mime_type,
        filename=audio_path.name
    )


@app.delete("/audio/{audio_id}")
async def delete_audio(audio_id: str):
    """Delete a generated audio file."""
    audio_path = DOWNLOADS_DIR / f"{audio_id}.wav"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    try:
        audio_path.unlink()
        return {"success": True, "message": "Audio deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Main =============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "init:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
