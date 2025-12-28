import os
import modal
import io
import torch
from pathlib import Path

# Setup Modal App
app = modal.App("phonex-worker")

# Define the persistent volume for models
models_volume = modal.Volume.from_name("phonex-models", create_if_missing=True)

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "txtsplit",
        "torch",
        "torchaudio",
        "cached_path",
        "transformers==4.27.4",
        "num2words==0.5.12",
        "unidic_lite==1.0.8",
        "unidic==1.1.0",
        "mecab-python3==1.0.9",
        "pykakasi==2.2.1",
        "fugashi==1.3.0",
        "g2p_en==2.1.0",
        "anyascii==0.3.2",
        "jamo==0.4.1",
        "gruut[de,es,fr]==2.2.3",
        "g2pkk>=0.1.1",
        "librosa==0.9.1",
        "pydub==0.25.1",
        "eng_to_ipa==0.0.2",
        "inflect==7.0.0",
        "unidecode==1.3.7",
        "pypinyin==0.50.0",
        "cn2an==0.5.22",
        "jieba==0.42.1",
        "langid==1.1.6",
        "tqdm",
        "loguru==0.7.2",
        "nltk",
        "openai-whisper",
        "whisper-timestamped",
        "wavmark>=0.0.3",
        "silero-vad>=5.1",
        "ffmpeg-python>=0.2.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.1",
        "numpy<2.0.0",
        "resemblyzer>=0.1.3"
    )
    .run_commands(
        "python -m unidic download",  # Download MeCab dictionary
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"",  # NLTK data
    )
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/models": models_volume},
    timeout=600,
    scaledown_window=60,
    max_containers=5  # Allow 5 parallel jobs for better concurrency
)
class VoiceEngine:
    @modal.enter()
    def init_models(self):
        """Load models once per container."""
        print("🚀 Initializing engines on GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Paths for models (inside volume)
        self.melotts_dir = Path("/models/melotts")
        self.openvoice_dir = Path("/models/open_voice")
        self.checkpoints_v2_dir = self.openvoice_dir / "checkpoints_v2"
        
        # Inject paths into sys.path
        import sys
        sys.path.insert(0, str(self.melotts_dir))
        sys.path.insert(0, str(self.openvoice_dir))
        
        # Initialize MeloTTS instances (caching)
        self.tts_models = {}
        
        # Initialize OpenVoice V2
        try:
            from openvoice.api import ToneColorConverter
            config_path = self.checkpoints_v2_dir / "converter" / "config.json"
            ckpt_path = self.checkpoints_v2_dir / "converter" / "checkpoint.pth"
            
            if config_path.exists() and ckpt_path.exists():
                self.converter = ToneColorConverter(str(config_path), device=self.device)
                self.converter.load_ckpt(str(ckpt_path))
                print("✅ OpenVoice V2 initialized")
            else:
                print("⚠️ OpenVoice checkpoints missing in /models")
                self.converter = None
        except Exception as e:
            print(f"❌ Failed to init OpenVoice: {e}")
            self.converter = None

    def get_tts_model(self, language: str):
        lang_upper = language.upper()
        if lang_upper not in self.tts_models:
            from melo.api import TTS
            self.tts_models[lang_upper] = TTS(language=lang_upper, device=self.device)
            print(f"✅ Loaded MeloTTS for {lang_upper}")
        return self.tts_models[lang_upper]

    @modal.method()
    def generate_tts(self, text: str, language: str, speaker_id: str, speed: float = 1.0) -> bytes:
        """Standard TTS generation - returns MP3."""
        from pydub import AudioSegment
        
        tts_model = self.get_tts_model(language)
        temp_wav = f"/tmp/tts_{os.urandom(8).hex()}.wav"
        temp_mp3 = f"/tmp/tts_{os.urandom(8).hex()}.mp3"
        
        melo_speaker_ids = tts_model.hps.data.spk2id
        if speaker_id not in melo_speaker_ids:
            raise ValueError(f"Invalid speaker {speaker_id}")
            
        tts_model.tts_to_file(
            text=text,
            speaker_id=melo_speaker_ids[speaker_id],
            output_path=temp_wav,
            speed=speed
        )
        
        # Convert WAV to MP3 for smaller file size
        audio = AudioSegment.from_wav(temp_wav)
        audio.export(temp_mp3, format="mp3", bitrate="128k")
        
        with open(temp_mp3, "rb") as f:
            data = f.read()
        
        for p in [temp_wav, temp_mp3]:
            if os.path.exists(p): os.remove(p)
        return data

    def split_text_into_chunks(self, text: str, max_chars: int = 1500):
        """Split text into manageable chunks by sentences."""
        import re
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i:i+max_chars].strip())
                    current_chunk = ""
                else:
                    current_chunk = sentence
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [c for c in chunks if c]

    @modal.method()
    def clone_voice(self, text: str, language: str, base_speaker: str, target_se_bytes: bytes, speed: float = 1.0, pitch: int = 0) -> bytes:
        """Voice cloning generation with chunking for long text - returns MP3."""
        from pydub import AudioSegment
        import gc
        
        if not self.converter:
            raise RuntimeError("ToneColorConverter not initialized")
        
        # Split long text into chunks to avoid OOM
        chunks = self.split_text_into_chunks(text, max_chars=1500)
        print(f"📝 Processing {len(chunks)} chunks...")
        
        # Parse language and accent (e.g., "EN-US" -> language="EN", accent="EN-US")
        lang_upper = language.upper()
        
        # Extract base language (for model loading) and speaker/accent
        if lang_upper.startswith("EN-"):
            base_lang = "EN"
            actual_speaker = lang_upper  # Use the specific accent (EN-US, EN-BR, etc.)
        else:
            base_lang = lang_upper
            # Map language to default speaker
            default_speakers = {
                "EN": "EN-US",
                "ES": "ES",
                "FR": "FR",
                "ZH": "ZH",
                "JP": "JP",
                "KR": "KR"
            }
            actual_speaker = default_speakers.get(base_lang, base_lang)
        
        tts_model = self.get_tts_model(base_lang)
        melo_speaker_ids = tts_model.hps.data.spk2id
        
        # Validate speaker exists for this language model
        if actual_speaker not in melo_speaker_ids:
            actual_speaker = list(melo_speaker_ids.keys())[0]
            print(f"⚠️ Using fallback speaker: {actual_speaker}")
        
        print(f"🎤 Using speaker: {actual_speaker} for language: {lang_upper}")
        
        # Pre-load speaker embeddings once (use the actual speaker key for SE file)
        speaker_key = actual_speaker.lower().replace("_", "-")
        se_path = self.checkpoints_v2_dir / "base_speakers" / "ses" / f"{speaker_key}.pth"
        if not se_path.exists():
            # Try with just the language code
            se_path = self.checkpoints_v2_dir / "base_speakers" / "ses" / f"{lang_upper.lower()}.pth"
        if not se_path.exists():
            raise ValueError(f"Base speaker SE not found for {actual_speaker}")
        source_se = torch.load(se_path, map_location=self.device)
        target_se = torch.load(io.BytesIO(target_se_bytes), map_location=self.device)
        
        # Process each chunk
        final_audio = AudioSegment.empty()
        
        for i, chunk_text in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk_text)} chars")
            
            temp_base = f"/tmp/base_{os.urandom(8).hex()}.wav"
            temp_clone = f"/tmp/clone_{os.urandom(8).hex()}.wav"
            
            try:
                # Generate base TTS
                tts_model.tts_to_file(
                    text=chunk_text,
                    speaker_id=melo_speaker_ids[actual_speaker],
                    output_path=temp_base,
                    speed=speed
                )
                
                # Convert voice
                self.converter.convert(
                    audio_src_path=temp_base,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=temp_clone,
                    message="Phonex"
                )
                
                # Append to final audio
                chunk_audio = AudioSegment.from_wav(temp_clone)
                final_audio += chunk_audio
                
            finally:
                for p in [temp_base, temp_clone]:
                    if os.path.exists(p): os.remove(p)
            
            # Clear GPU cache after each chunk
            torch.cuda.empty_cache()
            gc.collect()
        
        # Apply pitch shift if requested
        if pitch != 0:
            print(f"🎵 Applying pitch shift: {pitch} semitones")
            # Use pydub to shift pitch (by changing sample rate then resampling)
            original_sample_rate = final_audio.frame_rate
            # Pitch shift formula: new_rate = original_rate * (2 ** (semitones / 12))
            pitch_factor = 2 ** (pitch / 12.0)
            shifted_sample_rate = int(original_sample_rate * pitch_factor)
            
            # Change frame rate (this shifts pitch)
            pitched_audio = final_audio._spawn(final_audio.raw_data, overrides={
                "frame_rate": shifted_sample_rate
            })
            # Resample back to original rate (this preserves duration)
            final_audio = pitched_audio.set_frame_rate(original_sample_rate)
        
        # Export final audio as MP3
        temp_mp3 = f"/tmp/final_{os.urandom(8).hex()}.mp3"
        final_audio.export(temp_mp3, format="mp3", bitrate="128k")
        
        with open(temp_mp3, "rb") as f:
            result = f.read()
        
        os.remove(temp_mp3)
        print(f"✅ Generated {len(result)} bytes of audio")
        return result

@app.local_entrypoint()
def sync():
    """Sync local models folder to Modal volume."""
    print("🔄 Syncing models folder to Modal volume 'phonex-models'...")
    local_models_path = Path(__file__).parent / "models"
    
    if not local_models_path.exists():
        print(f"❌ Error: Local models folder not found at {local_models_path}")
        return

    # Use modal volume put to sync
    import subprocess
    # MSYS_NO_PATHCONV=1 is needed for Windows/Git Bash to avoid path conversion
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    
    try:
        subprocess.run(
            ["modal", "volume", "put", "phonex-models", str(local_models_path) + "/", "/"],
            check=True,
            env=env
        )
        print("✅ Sync completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Sync failed: {e}")

@app.local_entrypoint()
def main():
    """Testing entrypoint."""
    print("🚀 Modal Worker local entrypoint loaded.")
    print("Use 'modal run modal_worker.py::sync' to upload models.")
