#!/usr/bin/env python3
"""
Batch transcription CLI for MP4 files using Hugging Face Whisper models.

Usage example:
    python batch_trancribe.py --input-dir mp4_videos --audio-dir wav_audio --output-dir transcripts
"""

import os
import subprocess
import argparse
import sys
import traceback

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class VideoConverter:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def convert(self, input_path, output_path, overwrite=False):
        if os.path.exists(output_path) and not overwrite:
            print(f"🔁 Skipping conversion (exists): {output_path}")
            return
        command = [
            "ffmpeg", "-y" if overwrite else "-n", "-i", input_path,
            "-ar", str(self.target_sr), "-ac", "1", "-c:a", "pcm_s16le", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-medium", chunk_duration=30, device=None, language=None, translate=False, verbose=False):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.chunk_duration = chunk_duration
        self.language = language
        self.translate = translate
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(), 16000

    def _chunk_audio(self, waveform, sr):
        chunk_size = self.chunk_duration * sr
        return [waveform[i:i+chunk_size] for i in range(0, waveform.size(0), chunk_size)]

    def _transcribe_chunk(self, chunk):
        inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        forced_decoder_ids = None
        # If user requested translation or language forcing, try to obtain decoder prompt ids
        if self.language or self.translate:
            try:
                task = "translate" if self.translate else "transcribe"
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language or "en", task=task)
                self._log(f"Using forced_decoder_ids for language={self.language} task={task}")
            except Exception:
                # If the helper is not available or fails, fall back to None
                self._log("Could not set forced_decoder_ids, continuing without explicit language/task prompt.")

        output_ids = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    def transcribe(self, audio_path):
        waveform, sr = self._load_audio(audio_path)
        chunks = self._chunk_audio(waveform, sr)
        self._log(f"Transcribing {len(chunks)} chunks from {audio_path} (chunk_duration={self.chunk_duration}s)")
        texts = []
        for i, c in enumerate(chunks, start=1):
            try:
                self._log(f"  → chunk {i}/{len(chunks)}")
                texts.append(self._transcribe_chunk(c))
            except Exception:
                # Don't fail entire file on a single chunk issue
                self._log(f"Error transcribing chunk {i} of {audio_path}:\n{traceback.format_exc()}")
                texts.append("[ERROR: chunk failed]")
        return "\n\n".join(texts)

class BatchProcessor:
    def __init__(self, input_dir, audio_dir, output_dir, converter=None, transcriber=None, overwrite=False):
        self.input_dir = input_dir
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.converter = converter or VideoConverter()
        self.transcriber = transcriber
        self.overwrite = overwrite

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def process_all(self):
        mp4_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".mp4")]
        if not mp4_files:
            print("No .mp4 files found in", self.input_dir)
            return

        for fname in mp4_files:
            print(f"\n🎬 Processing {fname}")
            base = os.path.splitext(fname)[0]
            mp4_path = os.path.join(self.input_dir, fname)
            wav_path = os.path.join(self.audio_dir, f"{base}.wav")
            txt_path = os.path.join(self.output_dir, f"{base}.txt")

            try:
                self.converter.convert(mp4_path, wav_path, overwrite=self.overwrite)
            except subprocess.CalledProcessError:
                print(f"❌ ffmpeg failed for {mp4_path}, skipping.")
                continue
            except Exception:
                print(f"❌ Unexpected error converting {mp4_path}:")
                traceback.print_exc()
                continue

            try:
                text = self.transcriber.transcribe(wav_path)
            except Exception:
                print(f"❌ Transcription failed for {wav_path}:")
                traceback.print_exc()
                text = "[ERROR: transcription failed]"

            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Saved transcript: {txt_path}")
            except Exception:
                print(f"❌ Failed to write transcript {txt_path}:")
                traceback.print_exc()

def parse_args(argv):
    p = argparse.ArgumentParser(description="Batch transcribe MP4 videos using Whisper (Hugging Face).")
    p.add_argument("--input-dir", default="mp4_videos", help="Directory containing input .mp4 files")
    p.add_argument("--audio-dir", default="wav_audio", help="Directory to store intermediate .wav files")
    p.add_argument("--output-dir", default="transcripts", help="Directory to store transcripts (.txt)")
    p.add_argument("--model", default="openai/whisper-medium", help="Hugging Face model name (e.g. openai/whisper-small)")
    p.add_argument("--chunk-duration", type=int, default=30, help="Chunk duration in seconds")
    p.add_argument("--device", default=None, help="Device to run on (e.g. cpu or cuda). Auto-detect if omitted")
    p.add_argument("--language", default=None, help="Language code to force (e.g. 'en'). Optional.")
    p.add_argument("--translate", action="store_true", help="Force translation mode (task=translate)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files (ffmpeg -y)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)

    print("Device:", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Model:", args.model)
    print("Chunk duration (s):", args.chunk_duration)
    if args.language:
        print("Language:", args.language, " Translate mode:" , args.translate)

    try:
        transcriber = WhisperTranscriber(
            model_name=args.model,
            chunk_duration=args.chunk_duration,
            device=args.device,
            language=args.language,
            translate=args.translate,
            verbose=args.verbose
        )
    except Exception:
        print("Failed to load model:", args.model)
        traceback.print_exc()
        sys.exit(1)

    processor = BatchProcessor(
        input_dir=args.input_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        converter=VideoConverter(),
        transcriber=transcriber,
        overwrite=args.overwrite
    )
    processor.process_all()

if __name__ == "__main__":
    main(sys.argv[1:])
