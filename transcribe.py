"""
transcribe.py

Usage:
    python transcribe.py audiofile.~
"""

import sys
import argparse
from pathlib import Path

from openai_utils import transcribe, break_up_lines

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
    AUDIO_PATH = FILE_DIR / filename

    print("Selected audio file: ", AUDIO_PATH)

    if not AUDIO_PATH.exists():
        print(f"Error: Audio file not found: {AUDIO_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        # Transcription result, then break up lines by sentences
        result = break_up_lines(transcribe(AUDIO_PATH))        
        TRANSCRIPT_PATH = OUT_DIR / "transcription.txt"
        open(TRANSCRIPT_PATH, "w").write(result)
        print(f"Transcription saved to: {TRANSCRIPT_PATH}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    
    


if __name__ == "__main__":
    main()