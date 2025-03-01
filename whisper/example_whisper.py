import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from exla.models.whisper._implementations.whisper_cpu import Whisper_CPU
except ImportError:
    print("Error: Could not import Whisper_CPU.")
    print("Make sure the exla-sdk directory is in your Python path.")
    sys.exit(1)

def main():
    # Path to the audio file
    audio_path = 'data/speech.wav'
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        print("Please download a sample audio file using:")
        print("wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav -O data/speech.wav")
        sys.exit(1)
    
    # Initialize the Whisper model (using CPU implementation directly)
    model = Whisper_CPU(model_name='tiny.en')
    
    # Transcribe the audio
    result = model.transcribe(audio_path)
    
    # Print only the transcription
    print("\nTranscription:")
    print(result["text"])

if __name__ == "__main__":
    main() 