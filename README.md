
# python-audio-transcription

A small utility to batch-transcribe MP4 videos using OpenAI Whisper (via Hugging Face Transformers). The repository includes a simple pipeline that converts MP4 files to WAV using ffmpeg, loads audio with torchaudio, and transcribes using a Hugging Face Whisper model.

This README describes how to install, run, and customize the pipeline.

## Features
- Convert MP4 videos to 16 kHz mono WAV using ffmpeg
- Chunk long audio into configurable durations
- Transcribe audio using Hugging Face Whisper models (e.g., `openai/whisper-small`, `openai/whisper-medium`, `openai/whisper-large`)
- Save transcripts to text files (one text file per input MP4)

## Files
- `batch_trancribe.py` — main script that:
  - Converts `.mp4` files in `mp4_videos/` to `.wav` in `wav_audio/`
  - Transcribes WAV files and writes `.txt` transcripts to `transcripts/`
  - Note: filename contains a typo (`trancribe`) — the script name is preserved from the project.

## Requirements
- Python 3.8+
- ffmpeg (must be installed and available on PATH)
- PyTorch (CPU or GPU build)
- torchaudio
- transformers (Hugging Face)
- (optional) CUDA drivers if using a GPU

Example packages:
- torch
- torchaudio
- transformers

Install (example, adapt to your environment / PyTorch CUDA version):

```bash
# Using pip (adjust to your CUDA version; this is an example for CPU)
python -m pip install --upgrade pip
python -m pip install torch torchaudio transformers
```

If you use conda, prefer installing torch/torchaudio via the official instructions at https://pytorch.org to match your CUDA version.

Also install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from https://ffmpeg.org and add to PATH

## Usage

1. Place your `.mp4` files in the `mp4_videos/` directory (create it if missing).
2. Run the script:

```bash
python batch_trancribe.py
```

The script will:
- create `wav_audio/` and `transcripts/` if they don't exist
- convert each `.mp4` file to a 16 kHz mono WAV file
- chunk audio into pieces (default 30 seconds) and transcribe each chunk
- write a `.txt` file per input video into `transcripts/`

Default directories:
- input MP4s: `mp4_videos/`
- intermediate WAVs: `wav_audio/`
- transcripts: `transcripts/`

## Configuration / Customization

- Model selection:
  - The script uses the Hugging Face model `openai/whisper-medium` by default.
  - To use a different model, update the `WhisperTranscriber` constructor in `batch_trancribe.py`:
    - e.g., `"openai/whisper-small"`, `"openai/whisper-large"`

- Chunk duration:
  - Default chunk size is 30 seconds. Modify `chunk_duration` in `WhisperTranscriber(...)` to change it.

- GPU:
  - If CUDA is available and compatible PyTorch is installed, the script automatically uses GPU (`"cuda"`). Ensure your GPU and drivers match the installed torch build.

- Language / translation:
  - The example disables forced decoder/translation. If you want to force a language or translation, you can pass `forced_decoder_ids` or set `task="translate"` and `language="en"` using Hugging Face `generate()` options; be mindful of model capabilities.

## Notes and tips
- Large Whisper models require more GPU memory; use smaller models for CPU or limited-memory environments.
- ffmpeg conversion uses `pcm_s16le` and resamples to 16000 Hz mono — this is what Whisper expects.
- The script currently has no CLI flags — to make it more flexible, consider adding argparse to accept:
  - input/output directory arguments
  - model selection
  - chunk duration
  - concurrency / batching options
- Error handling: subprocess calls to ffmpeg use `check=True`, which will raise an exception on failure. Consider adding try/except to continue processing other files if one fails.

## Example improvements (suggestions)
- Add a CLI (`argparse`) to configure directories, model, chunk size, and other options.
- Add progress logging and per-chunk timestamps.
- Add parallel processing for conversions/transcriptions (be careful with GPU memory).
- Add unit tests and CI checks.

## License
Add a license of your choice (e.g., MIT) by including a `LICENSE` file.

## Acknowledgements
- Uses Hugging Face Transformers and the Whisper model.
- Uses torchaudio for loading/resampling audio.
- Uses ffmpeg for media conversion.
