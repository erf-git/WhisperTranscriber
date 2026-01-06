import whisper
model = whisper.load_model("base")
result = model.transcribe("Recording(3).m4a", fp16=False)
print(result["text"])