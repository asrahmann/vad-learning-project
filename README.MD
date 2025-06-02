# Real-Time Voice Activity Detection (VAD) System

## Overview

This project implements a real-time Voice Activity Detection (VAD) system using Python. It listens to an audio stream from a selected microphone, detects speech segments, and saves them as WAV files. The core VAD functionality is powered by the Silero VAD model, accessed via PyTorch Hub.

This system is designed to be relatively robust, processing audio in small chunks and using Silero's `VADIterator` for stateful speech detection to accurately capture utterances.

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