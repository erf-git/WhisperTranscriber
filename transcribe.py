"""
transcribe.py

Usage:
    python transcribe.py audiofile.~
"""

import sys
import argparse
from pathlib import Path

from whisper_utils import recorded_transcribe

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
        result = recorded_transcribe(audio_file_path)        
        open(OUT_DIR / "transcription.txt", "w").write(result)
        print("Transcription complete.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()