#!/usr/bin/env python3
"""
Convert existing WAV files in downloads folder to MP3.
Run this once on the server to migrate old files.
"""

import os
from pathlib import Path
from pydub import AudioSegment

DOWNLOADS_DIR = Path(__file__).parent / "downloads"

def convert_wav_to_mp3():
    """Convert all WAV files to MP3 and delete originals."""
    wav_files = list(DOWNLOADS_DIR.glob("*.wav"))
    
    if not wav_files:
        print("No WAV files found to convert.")
        return
    
    print(f"Found {len(wav_files)} WAV files to convert...")
    
    for wav_path in wav_files:
        mp3_path = wav_path.with_suffix(".mp3")
        
        # Skip if MP3 already exists
        if mp3_path.exists():
            print(f"  Skip: {wav_path.name} (MP3 already exists)")
            continue
        
        try:
            print(f"  Converting: {wav_path.name}...", end=" ")
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(mp3_path), format="mp3", bitrate="128k")
            
            # Delete original WAV
            os.remove(wav_path)
            print("Done!")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    convert_wav_to_mp3()
