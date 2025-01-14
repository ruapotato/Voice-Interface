#!/usr/bin/env python3

import torch
import pyaudio
import wave
import numpy as np
import os
import sys
from datetime import datetime
import time
from collections import deque
import requests
import json
import argparse
import signal
import logging
import queue
from threading import Thread, Event, Lock
from typing import List, Dict, Any, Optional, Tuple

CHUNK_SIZE = 512
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
TTS_RATE = 24000
SILENCE_LIMIT = 0.7
PREV_AUDIO_SECONDS = 0.5
MIN_SILENCE_DETECTIONS = 3
MIN_AUDIO_DURATION = 0.35
VAD_THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_playing = Lock()

def init_models():
    cache_dir = os.path.expanduser('~/.cache/silero')
    tts_path = os.path.join(cache_dir, 'v3_en.pt')
    os.makedirs(cache_dir, exist_ok=True)
    
    torch.set_num_threads(1)
    vad_model, *_ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    
    stt_model, decoder, _ = torch.hub.load(
        'snakers4/silero-models', 'silero_stt', language='en', device=DEVICE
    )
    
    if not os.path.exists(tts_path):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt', tts_path)
    tts_model = torch.package.PackageImporter(tts_path).load_pickle("tts_models", "model")
    
    vad_model.to(DEVICE)
    stt_model.to(DEVICE)
    tts_model.to(DEVICE)
    
    return vad_model, stt_model, decoder, tts_model

def get_ai_response(text: str) -> str:
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': f"You are a helpful voice assistant. Be concise. User says: {text}",
                'stream': False
            },
            timeout=30
        )
        return response.json().get('response', '')
    except Exception as e:
        return f"Failed to get response: {e}"

def play_audio(audio: torch.Tensor, pa: pyaudio.PyAudio):
    if audio is None:
        return
        
    try:
        with is_playing:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=TTS_RATE,
                output=True
            )
            stream.write(audio.numpy().tobytes())
            stream.stop_stream()
            stream.close()
    except Exception as e:
        print(f"Audio playback failed: {e}")

def process_speech_segment(audio_chunks: List[bytes], models: Tuple, pa: pyaudio.PyAudio):
    vad_model, stt_model, decoder, tts_model = models
    
    if not audio_chunks:
        return
        
    print("\r🤔 Processing...", end="", flush=True)
    
    audio_data = np.concatenate([np.frombuffer(chunk, dtype=np.float32) for chunk in audio_chunks])
    duration = len(audio_data) / RATE
    
    if duration < MIN_AUDIO_DURATION:
        return
    
    wav_tensor = torch.from_numpy(audio_data).to(DEVICE).unsqueeze(0)
    with torch.inference_mode():
        emission = stt_model(wav_tensor)
        transcription = decoder(emission[0].cpu())
    
    if not transcription:
        return
        
    print(f"\n🗣️ You: {transcription}")
    
    response = get_ai_response(transcription)
    print(f"🤖 Assistant: {response}")
    
    try:
        audio = tts_model.apply_tts(
            text=response,
            speaker='en_0',
            sample_rate=TTS_RATE
        )
        play_audio(audio, pa)
    except Exception as e:
        print(f"Speech synthesis failed: {e}")
    
    print("\r🎤 Ready...", end="", flush=True)

def process_audio(audio_buffer: queue.Queue, models: Tuple, pa: pyaudio.PyAudio, running: Event):
    vad_model = models[0]
    audio_chunks = []
    silence_chunks = 0
    is_speaking = False
    prev_frames = deque(maxlen=int(PREV_AUDIO_SECONDS * RATE / CHUNK_SIZE))
    
    while running.is_set():
        try:
            chunk = audio_buffer.get(timeout=0.1)
            
            # Skip VAD if currently playing audio
            if is_playing.locked():
                continue
                
            audio_data = np.frombuffer(chunk, dtype=np.float32).copy()
            audio_tensor = torch.from_numpy(audio_data).to(DEVICE)
            
            if len(audio_data) != CHUNK_SIZE:
                continue
            
            speech_prob = vad_model(audio_tensor.unsqueeze(0), RATE).item()
            
            if speech_prob >= VAD_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    print("\r🎙️ Listening...", end="", flush=True)
                    audio_chunks.extend(list(prev_frames))
                
                audio_chunks.append(chunk)
                silence_chunks = 0
            else:
                if is_speaking:
                    silence_chunks += 1
                    audio_chunks.append(chunk)
                    
                    if (silence_chunks * CHUNK_SIZE / RATE >= SILENCE_LIMIT and
                        silence_chunks >= MIN_SILENCE_DETECTIONS):
                        process_speech_segment(audio_chunks, models, pa)
                        audio_chunks = []
                        silence_chunks = 0
                        is_speaking = False
                else:
                    prev_frames.append(chunk)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in processing: {e}")

def audio_callback(in_data, frame_count, time_info, status, audio_buffer: queue.Queue, running: Event):
    if not running.is_set():
        return (None, pyaudio.paComplete)
    audio_buffer.put(in_data)
    return (in_data, pyaudio.paContinue)

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s', stream=sys.stderr)
    
    running = Event()
    running.set()
    audio_buffer = queue.Queue()
    
    models = init_models()
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=lambda *args: audio_callback(*args, audio_buffer, running)
    )
    
    processing_thread = Thread(
        target=process_audio,
        args=(audio_buffer, models, pa, running),
        daemon=True
    )
    
    print("🎤 Voice Assistant Ready (Ctrl+C to exit)")
    print(f"Using device: {DEVICE}")
    processing_thread.start()
    
    try:
        while running.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        running.clear()
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()
