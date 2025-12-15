
import subprocess
import wave
import os
import json
import time

import simpleaudio as sa
import sounddevice as sd
import numpy as np
import requests
from faster_whisper import WhisperModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Assumes the script is run from the same directory containing 'piper_env'
# VOICES_DIR = os.path.join("piper_env", "piper_voices")
VOICES_DIR = os.path.join("piper_env")
RAG_SERVER_URL = "http://127.0.0.1:8000/ask"

# ASR Model (Transcription)
ASR_MODEL_SIZE = "small" # Use "medium" for better accuracy if VRAM allows
try:
    print(f"Loading ASR model ({ASR_MODEL_SIZE})...")
    # Setting device="cpu" is robust for most machines
    asr = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    asr = None

# TTS Model (Piper) selection
def get_voice_paths(language: str):
    """Returns the .onnx and .json paths for the chosen voice."""
    if language.lower() == "en":
        voice_name = "en_US-lessac-high"
    elif language.lower() == "de":
        voice_name = "de_DE-thorsten-high" # Higher quality German voice
    else:
        raise ValueError("Unsupported language for TTS.")

    model_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx")
    json_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx.json")

    # Check if files exist (critical step)
    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print("-" * 50)
        print(f"ERROR: Voice files for '{voice_name}' not found!")
        print(f"Expected model at: {model_path}")
        print("Please ensure you have downloaded and placed the .onnx and .onnx.json files.")
        print("-" * 50)
        return None, None

    return model_path, json_path

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def record(seconds: int = 6, samplerate: int = 16000) -> tuple[np.ndarray, int]:
    """Records audio from the default microphone."""
    print("🎤 Recording... (Speak now)")
    # Using a high sample rate for better quality
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("🛑 Done recording.")
    return (audio.flatten(), samplerate)

def transcribe(audio: np.ndarray, samplerate: int, language: str = None) -> str:
    """Transcribes audio using Faster-Whisper."""
    if asr is None:
        return "Transcription service is unavailable."

    print("🧠 Transcribing audio...")
    segments, info = asr.transcribe(audio, language=language, beam_size=5)
    text = "".join([s.text for s in segments])
    return text.strip()

def tts_piper(text: str, language: str, out_wav: str = "answer.wav") -> str:
    """Generates speech using the local Piper CLI."""
    model_path, json_path = get_voice_paths(language)

    if not model_path:
        return "" # Return empty string on failure

    # Assumes 'piper' executable is in your PATH (e.g., inside piper_env/bin)
    cmd = ["piper", "-m", model_path, "-c", json_path, "-f", out_wav]

    print(f"🗣️ Generating speech using Piper...")
    try:
        # Using subprocess.run for simpler execution handling
        result = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, check=True)
        # print(f"Piper stderr: {result.stderr.decode()}") # Uncomment for debugging
        return out_wav
    except subprocess.CalledProcessError as e:
        print(f"Error calling Piper CLI: {e}")
        print(f"Command: {' '.join(cmd)}")
        return ""
    except FileNotFoundError:
        print("Error: Piper executable not found. Ensure your environment is active.")
        return ""

def play_wav(path: str):
    """Plays a WAV file using simpleaudio."""
    if not path or not os.path.exists(path):
        return

    try:
        with wave.open(path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            # Play buffer: data, num_channels, bytes_per_sample, sample_rate
            play_obj = sa.play_buffer(
                data,
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate()
            )
            play_obj.wait_done()
    except Exception as e:
        print(f"Error during audio playback: {e}")

# ==============================================================================
# MAIN VOICE INTERACTION LOOP
# ==============================================================================

def ask_voice(language: str = "en"):
    """
    Main function to handle the full voice-to-voice RAG interaction.
    """
    if asr is None:
        print("Cannot proceed: ASR model failed to load.")
        return

    # ASR Language Hint (e.g., "en" or "de")
    asr_language_hint = "en" if language == "en" else "de"

    audio, sr = record(seconds=10)
    text = transcribe(audio, sr, language=asr_language_hint)

    if not text:
        print("No text transcribed. Please try again.")
        return

    print(f"\n🙋 You: {text}")

    # 2. Call the RAG server
    try:
        print(f"🌐 Querying RAG server at {RAG_SERVER_URL}...")
        resp = requests.post(
            RAG_SERVER_URL,
            json={"question": text, "language": language},
            timeout=60
        )
        resp.raise_for_status() # Check for HTTP errors
        ans = resp.json()["answer"]

    except requests.exceptions.RequestException as e:
        ans = f"Error: Could not connect to the RAG server. Please ensure the server is running on {RAG_SERVER_URL}. ({e})"
        print(f"\n🤖 Tutor: {ans}")
        tts_piper(ans, "en") # Use English voice for error messages
        return

    # 3. Text-to-Speech and Playback
    print(f"🤖 Tutor: {ans}")
    wav = tts_piper(ans, language)

    if wav:
        play_wav(wav)
        os.remove(wav) # Clean up the temporary WAV file

# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Example usage:
    # To run in German: ask_voice(language="de")
    # To run in English (default):
    ask_voice(language="en")
