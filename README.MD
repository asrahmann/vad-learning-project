# Real-Time Voice Activity Detection (VAD) System

## Overview

This project implements a real-time Voice Activity Detection (VAD) system using Python. It listens to an audio stream from a selected microphone, detects speech segments, and saves them as WAV files. The core VAD functionality is powered by the Silero VAD model, accessed via PyTorch Hub.

This system is designed to be relatively robust, processing audio in small chunks and using Silero's `VADIterator` for stateful speech detection to accurately capture utterances.

## Key Learnings

This project was undertaken specifically to understand and implement Voice Activity Detection. The process provided several key insights:

* **VAD Parameter Sensitivity:** Learned that the performance of a VAD system is highly sensitive to its configuration parameters. Small adjustments to settings like `threshold` (for speech detection sensitivity), `min_silence_duration_ms` (to define the end of an utterance), and `speech_pad_ms` (to avoid clipping words) can drastically alter the accuracy and perceived quality of speech segmentation.
* **Challenges in Real-Time Audio Streaming:** Discovered the complexities of processing audio in real-time. Initial approaches using simple blocking reads (`sd.rec()`) for larger audio chunks proved insufficient for continuous and accurate VAD, leading to issues like data loss or "choppy" audio. The transition to `sd.InputStream` and a more continuous processing loop was crucial.
* **Importance of Model-Specific Input Requirements:** A critical lesson was the necessity of adhering to the precise input specifications of the underlying machine learning model. The Silero VAD model requires audio chunks of a specific size (512 samples at 16kHz). Feeding it incorrectly sized chunks led to errors and suboptimal performance. Correcting this was a turning point for the project's success.
* **Stateful vs. Stateless VAD:** Understood the limitations of stateless VAD (applying detection on isolated chunks) for continuous speech. The adoption of Silero's `VADIterator`, which maintains an internal state across chunks, provided a much more robust and natural segmentation of speech, better handling pauses and variations within an utterance.
* **Iterative Debugging for Complex Audio Issues:** Realized that problems like "choppy audio" or "missing phonemes" can stem from various sources (VAD logic, data loss, model input errors). A systematic, iterative debugging process—hypothesizing, implementing changes, observing results, and refining—was essential to diagnose and resolve these issues effectively.
* **Leveraging Specialized Libraries/Utilities:** Appreciated the value of using tools designed for specific tasks. `VADIterator` abstracted away much of the complex state management and timing logic required for high-quality streaming VAD, leading to a cleaner and more effective implementation.
* **Practical Application of Audio Processing Concepts:** Gained hands-on experience with fundamental audio processing concepts like sample rates, chunking, gain adjustment, and data type conversions (float32 to int16 for WAV).


**Core Technologies:**
* Python 3
* Silero VAD (via PyTorch Hub)
* PyTorch
* SoundDevice
* NumPy

## Features

* Real-time voice activity detection from a microphone input.
* Uses Silero's `VADIterator` for robust, streaming speech segment detection.
* Automatically saves detected speech segments to `.wav` files with timestamps.
* Configurable parameters for VAD sensitivity, silence duration, gain, and speech padding.
* Microphone selection from available input devices.
* Includes overflow detection for the input audio stream to help diagnose capture issues.

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.8 or newer.
    * `pip` for package installation.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv311 # Or your preferred venv name
    source venv311/bin/activate # On Linux/macOS
    venv311\Scripts\activate # On Windows
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file (you can generate one in your active environment using `pip freeze > requirements.txt`). Then run:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include `torch`, `torchaudio`, `sounddevice`, and `numpy`. Silero VAD will be downloaded via `torch.hub` on the first run if not cached.

## Usage

Run the script from the command line:

```bash
python vad_iterator_recorder.py [OPTIONS]