import sys
import os

from inferx.models.whisper import whisper


model = whisper()

# Path to the audio file
audio_path = 'data/speech.wav'

# Check if the audio file exists
if not os.path.exists(audio_path):
    print(f"Error: Audio file '{audio_path}' not found.")
    print("Please download a sample audio file using:")
    print("wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav -O data/speech.wav")
    sys.exit(1)

# Transcribe the audio
result = model.transcribe(audio_path)

# Print only the transcription
print("\nTranscription:")
print(result["text"])
