# Voice Interface

A real-time voice assistant using Silero models for speech-to-text, voice activity detection, and text-to-speech. This is a simple demo demonstrating how easy it is to create a voice assistant these days.

## Features

- Real-time voice activity detection
- Speech-to-text transcription
- Text-to-speech synthesis
- GPU acceleration support

## Requirements

- Python 3.8+
- PyTorch
- PyAudio
- NumPy
- Local Ollama instance running Mistral

## Installation

```bash
# Install required packages
pip install torch numpy pyaudio requests

# Clone and run
git clone https://github.com/ruapotato/voice_interface.git
cd voice_interface
python voice_assistant.py
```

## Usage

Basic:
```bash
python voice_assistant.py
```

## License

GPL3 © David Hamner
See https://github.com/snakers4/silero-models/blob/master/LICENSE as it is a base dependency.

## Technical Details

- Uses Silero VAD for voice activity detection
- Uses Silero STT for speech-to-text
- Uses Silero TTS for text-to-speech
- Uses Ollama Mistral for text responses
- Audio captured at 16kHz, TTS output at 24kHz
- Auto-detects CUDA support
