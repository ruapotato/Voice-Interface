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

def format_for_speech(text: str) -> str:
    num_dict = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '25': 'twenty five',
        '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty',
        '70': 'seventy', '80': 'eighty', '90': 'ninety', '100': 'one hundred',
        '1000': 'one thousand'
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            if word in num_dict:
                words[i] = num_dict[word]
            else:
                words[i] = ' '.join(num_dict[digit] for digit in word)
    
    text = ' '.join(words)
    
    replacements = {
        '%': ' percent',
        '$': ' dollars',
        '&': ' and',
        '+': ' plus',
        '=': ' equals',
        '-': ' minus',
        '*': ' times',
        '/': ' divided by',
        '<': ' less than',
        '>': ' greater than',
    }
    
    for symbol, replacement in replacements.items():
        text = text.replace(symbol, replacement)
    
    return text

def init_models():
    print("Initializing models... ", end="", flush=True)
    
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
    
    print("Done!", flush=True)
    return vad_model, stt_model, decoder, tts_model

def get_ai_response(text: str) -> str:
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': f"Reply with a single words or single sentences, don't use lists. User says: {text}",
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
        
    is_playing.acquire()
    try:
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
    finally:
        is_playing.release()

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
        if len(response.strip()) > 0:
            formatted_response = format_for_speech(response)
            audio = tts_model.apply_tts(
                text=formatted_response,
                speaker='en_0',
                sample_rate=TTS_RATE
            )
            play_audio(audio, pa)
    except Exception as e:
        print(f"Speech synthesis failed: {str(e)}")
    
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
    if not is_playing.locked():
        audio_buffer.put(in_data)
    return (in_data, pyaudio.paContinue)

def list_audio_devices():
    pa = pyaudio.PyAudio()
    info = []
    for i in range(pa.get_device_count()):
        try:
            info.append(pa.get_device_info_by_index(i))
        except OSError:
            continue
    pa.terminate()
    return info

def main():
    parser = argparse.ArgumentParser(description='Voice Assistant')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    parser.add_argument('--input-device', type=int, help='Input device index')
    parser.add_argument('--output-device', type=int, help='Output device index')
    args = parser.parse_args()
    
    if args.list_devices:
        devices = list_audio_devices()
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']}")
            print(f"   Input channels: {dev['maxInputChannels']}")
            print(f"   Output channels: {dev['maxOutputChannels']}")
        return
    
    level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s', stream=sys.stderr)
    
    running = Event()
    running.set()
    audio_buffer = queue.Queue()
    
    models = init_models()
    
    pa = pyaudio.PyAudio()
    input_params = {
        'format': FORMAT,
        'channels': CHANNELS,
        'rate': RATE,
        'input': True,
        'frames_per_buffer': CHUNK_SIZE,
        'stream_callback': lambda *args: audio_callback(*args, audio_buffer, running)
    }
    
    if args.input_device is not None:
        input_params['input_device_index'] = args.input_device
    
    if args.output_device is not None:
        input_params['output_device_index'] = args.output_device
    
    stream = pa.open(**input_params)
    
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
