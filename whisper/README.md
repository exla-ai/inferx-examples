# Whisper Speech Recognition Example

This example demonstrates how to use the Whisper model for speech recognition and transcription.

## Overview

[Whisper](https://github.com/openai/whisper) is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It is designed to transcribe speech in multiple languages.

## Prerequisites

- EXLA SDK installed
- Audio file for transcription

## Sample Audio

If you don't have an audio file to test with, you can download a sample file:

```bash
mkdir -p data
wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav -O data/speech.wav
```

## Usage

To transcribe an audio file:

```bash
python example_whisper.py
```

This will:
1. Load the Whisper model
2. Transcribe the audio file at `data/speech.wav`
3. Print the transcription 