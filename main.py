"""
main.py

Usage:
    python main.py audiofile.txt
"""

import whisper
import librosa
import sys
import argparse
import os
from pathlib import Path

FILE_DIR = None # Global variable to store working file directory

def main() -> None: 
    global FILE_DIR
    FILE_DIR = Path.cwd() # Set working file directory
    
    OUT_DIR = FILE_DIR / "out"
    OUT_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    parser = argparse.ArgumentParser(
        description="Provde audio file name to transcribe."
    )
    parser.add_argument("filename", help="Name of the audio file")

    args = parser.parse_args()
    filename = args.filename
    audio_file_path = FILE_DIR / "audio" / filename

    print("Selected audio file: ", audio_file_path)

    if not audio_file_path.exists():
        print(f"Error: Audio file not found: {audio_file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe_audio(audio_file_path)        
        TRANSCRIPT_PATH = OUT_DIR / "transcription.txt"
        open(TRANSCRIPT_PATH, "w").write(result)
        print(f"Transcription saved to: {TRANSCRIPT_PATH}")
        
        # Generate summary
        print("Generating summary...")
        summary = summarize_text(result)
        SUMMARY_PATH = OUT_DIR / "summary.txt"
        open(SUMMARY_PATH, "w").write(summary)
        print(f"Summary saved to: {SUMMARY_PATH}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

def transcribe_audio(file_path: Path) -> str:
    model = whisper.load_model("base")

    audio, sr = librosa.load(str(file_path), sr=16000) # Load audio file with 16000 Hz sampling rate with Librosa

    audio = whisper.pad_or_trim(audio) # Process audio through whisper
    mel = whisper.log_mel_spectrogram(audio).to(model.device) # Convert audio to mel spectrogram

    options = whisper.DecodingOptions(fp16=False) # Set decoding options
    result = whisper.decode(model, mel, options) # Decode the audio
    return result.text # Return the transcription


if __name__ == "__main__":
    main()