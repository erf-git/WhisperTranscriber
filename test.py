from pathlib import Path
from openai_utils import transcribe, break_up_lines
from bart_utils import summarize

with open("C:/Users/uneth/Downloads/Coding/WhisperTranscriber/out/test.txt", 'r') as f:
    content = f.read()

SUMMARY_PATH = "C:/Users/uneth/Downloads/Coding/WhisperTranscriber/out/summary.txt"
summary = summarize(content)
open(SUMMARY_PATH, "w").write(summary)
print(f"Summary saved to: {SUMMARY_PATH}")