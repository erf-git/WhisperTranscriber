# WhisperTranscriber
Live and pre-recorded audio transcription with OpenAI Whisper, with automatic summarization using LLM

## Features
- Transcribe audio files using OpenAI Whisper
- Automatically generate summaries of transcribed content using LLM (OpenAI GPT-3.5-turbo)
- Fallback to simple extractive summary if LLM API is not available

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Setup

For LLM-based summarization, set your OpenAI API key as an environment variable:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

If you don't set the API key, the tool will use a simple extractive summary (first few sentences) as a fallback.

## Usage

```bash
python main.py <audio_filename>
```

Example:
```bash
python main.py Recording.mp3
```

The script will:
1. Transcribe the audio file and save it to `out/transcription.txt`
2. Generate a summary and save it to `out/summary.txt`

## Output Files

- `out/transcription.txt` - Full transcription of the audio
- `out/summary.txt` - Summary of the transcribed content
