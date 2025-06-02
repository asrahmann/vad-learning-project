import torch
import numpy as np
import sounddevice as sd
import time
import argparse
import os
from scipy.io.wavfile import write
from datetime import datetime

# ------------------ Argument Parser ------------------
parser = argparse.ArgumentParser(
    description="Streaming Silero VAD recorder using VADIterator. \n"
                "Automatically detects speech segments and saves them. \n"
                "Ensure system microphone volume is adequate. Use --gain for additional amplification.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--threshold', type=float, default=0.5, help='VAD sensitivity threshold (0.0 to 1.0). Lower is more sensitive. Silero VAD default is often 0.5.')
parser.add_argument('--gain', type=float, default=1.0, help='Amplification factor for microphone input. This affects VAD and saved audio.')
parser.add_argument('--min_silence_ms', type=int, default=700, help='Minimum duration of silence (in ms) after speech to consider the utterance ended.')
parser.add_argument('--speech_pad_ms', type=int, default=200, help='Padding (in ms) added to the start and end of detected speech segments.')
# The --min_speech_chunk_ms argument is kept in argparse for now, but not directly used in VADIterator constructor.
# VADIterator uses internal logic and other params for minimum speech detection.
parser.add_argument('--min_speech_chunk_ms', type=int, default=150, help='(Informational) Conceptual minimum duration of a speech chunk (in ms). Not directly passed to VADIterator __init__.')
parser.add_argument('--device_index', type=int, default=None, help='Manually specify the input device index.')


args = parser.parse_args()

# ------------------ Constants and Setup ------------------
SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
# CRITICAL FIX: Silero VAD model (when called directly by VADIterator) expects 512 samples for 16kHz
CHUNK_SAMPLES = 512
CHUNK_DURATION_MS = int((CHUNK_SAMPLES / SAMPLE_RATE) * 1000) # Should be 32 ms
SAVE_DIR = "vad_iterator_recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------ Load Silero VAD Model and Utilities ------------------
print("Loading Silero VAD model...")
try:
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, VADIterator, _) = utils
except Exception as e:
    print(f"Error loading Silero VAD model: {e}")
    exit()
print("Silero VAD model loaded.")

# ------------------ Microphone Selection ------------------
if args.device_index is None:
    print("\n🎤 Available input devices:")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        exit()

    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if not input_devices:
        print("No input devices found.")
        exit()

    for i, dev in enumerate(input_devices):
        print(f"{i}: {dev['name']}")

    choice = -1
    while choice < 0 or choice >= len(input_devices):
        try:
            choice_str = input(f"\nSelect input device index (0 to {len(input_devices)-1}): ")
            choice = int(choice_str)
            if not (0 <= choice < len(input_devices)):
                print("Invalid selection.")
                choice = -1
        except ValueError:
            print("Invalid input. Please enter a number.")
            choice = -1
    selected_device_index = input_devices[choice]['index']
    selected_device_name = input_devices[choice]['name']
else:
    try:
        selected_device_info = sd.query_devices(args.device_index)
        selected_device_index = args.device_index
        selected_device_name = selected_device_info['name']
        if selected_device_info['max_input_channels'] == 0:
            print(f"Error: Device {selected_device_index} ({selected_device_name}) has no input channels.")
            exit()
    except Exception as e:
        print(f"Error querying specified device index {args.device_index}: {e}")
        exit()


# ------------------ Initialize VADIterator ------------------
vad_iterator = VADIterator(
    model,
    threshold=args.threshold,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=args.min_silence_ms,
    speech_pad_ms=args.speech_pad_ms
    # min_speech_duration_ms was removed as it caused a TypeError with VADIterator constructor
)
print("\nVADIterator initialized.")
print(f"Listening on: {selected_device_name} (Index: {selected_device_index})")
print(f"Chunk size: {CHUNK_SAMPLES} samples ({CHUNK_DURATION_MS}ms)") # Will now be 512 samples, 32ms
print(f"VAD Threshold: {args.threshold} | Gain: {args.gain}")
print(f"Min Silence Duration: {args.min_silence_ms}ms | Speech Pad: {args.speech_pad_ms}ms") # Removed min_speech_chunk_ms from this print
print(f"Recordings will be saved in '{os.path.abspath(SAVE_DIR)}'")
print("\nIMPORTANT: Ensure your SYSTEM microphone volume is adequately set for best results.")
print("Starting audio stream and VAD...")

current_recording_chunks = []
recording_active = False

# ------------------ Main VAD Loop ------------------
try:
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype='float32',
                        blocksize=CHUNK_SAMPLES, # Now 512
                        device=selected_device_index) as stream:
        print("\n🎧 Audio stream started. Listening for speech...")
        while True:
            audio_chunk_np, overflowed = stream.read(CHUNK_SAMPLES)
            if overflowed:
                print("⚠️ WARNING: Input audio buffer overflowed! Some audio data may have been lost.")

            processed_chunk_np = audio_chunk_np * args.gain
            processed_chunk_np = np.clip(processed_chunk_np, -1.0, 1.0)
            
            audio_tensor_for_vad = torch.from_numpy(processed_chunk_np.flatten()).float()

            speech_dict = vad_iterator(audio_tensor_for_vad, return_seconds=False)

            # rms = np.sqrt(np.mean(processed_chunk_np**2))
            # print(f"   [Chunk] RMS (post-gain): {rms:.4f}") # Optional: very verbose for 32ms chunks

            if speech_dict:
                if 'start' in speech_dict:
                    print(f"🎤 Voice detected (segment start at sample {speech_dict['start']})")
                    if not recording_active:
                        current_recording_chunks = []
                        recording_active = True
                
            if recording_active:
                current_recording_chunks.append(processed_chunk_np.flatten())

            if speech_dict and 'end' in speech_dict:
                print(f"🎤 Voice segment ended (at sample {speech_dict['end']}).")
                if recording_active and current_recording_chunks:
                    recording_active = False 
                    print("💾 Processing and saving recording...")
                    
                    merged_audio = np.concatenate(current_recording_chunks)
                    
                    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    merged_audio_int16 = (merged_audio * 32767).astype(np.int16)
                    
                    try:
                        write(filepath, SAMPLE_RATE, merged_audio_int16)
                        duration_sec = len(merged_audio) / SAMPLE_RATE
                        print(f"💾 Saved: {filepath} ({duration_sec:.2f} seconds)")
                    except Exception as e:
                        print(f"Error saving audio file: {e}")
                    
                    current_recording_chunks = [] 
                else:
                    if not current_recording_chunks and recording_active:
                         print("INFO: 'end' signal received but no chunks recorded. Speech may have been too short or parameters too strict.")
                    recording_active = False 

                vad_iterator.reset_states() 
                print("\n🎧 Listening for new speech...")

except KeyboardInterrupt:
    print("\n🛑 Recording stopped by user.")
    if recording_active and current_recording_chunks: 
        print("💾 Processing and saving final clip due to user interruption...")
        merged_audio = np.concatenate(current_recording_chunks)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_interrupt.wav")
        filepath = os.path.join(SAVE_DIR, filename)
        merged_audio_int16 = (merged_audio * 32767).astype(np.int16)
        try:
            write(filepath, SAMPLE_RATE, merged_audio_int16)
            duration_sec = len(merged_audio) / SAMPLE_RATE
            print(f"💾 Final clip saved: {filepath} ({duration_sec:.2f} seconds)")
        except Exception as e:
            print(f"Error saving final audio clip: {e}")
    else:
        print("No active recording to save or recording list is empty.")
except Exception as e:
    print(f"\nAn unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Exiting VAD recorder application.")