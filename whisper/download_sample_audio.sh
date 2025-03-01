#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download sample audio file
echo "Downloading sample audio file..."
wget https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav -O data/speech.wav

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Sample audio file downloaded successfully to data/speech.wav"
    echo "You can now run the example with: python example_whisper.py"
else
    echo "Error downloading sample audio file"
    exit 1
fi 