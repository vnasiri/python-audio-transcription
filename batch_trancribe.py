import os
import subprocess
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 🎥 Convert .mp4 to .wav
class VideoConverter:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def convert(self, input_path, output_path):
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(self.target_sr), "-ac", "1", "-c:a", "pcm_s16le", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 🧠 Load model and processor
class WhisperTranscriber:
    # Can be: whisper-small, medium, large
    def __init__(self, model_name="openai/whisper-medium", chunk_duration=30):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.chunk_duration = chunk_duration

    # 🎧 Load and resample audio
    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(), 16000
    
    # ✂️ Split long audio into chunks
    def _chunk_audio(self, waveform, sr):
        chunk_size = self.chunk_duration * sr
        return [waveform[i:i+chunk_size] for i in range(0, waveform.size(0), chunk_size)]

    # 🌍 Detect language from first chunk
    def _transcribe_chunk(self, chunk):
        inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output_ids = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=None,  # Explicitly set forced_decoder_ids to None
            # language="en",  # Ensure translation to English
            # task="translate"
        )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 📜 Transcribe single chunk
    def transcribe(self, audio_path):
        waveform, sr = self._load_audio(audio_path)
        chunks = self._chunk_audio(waveform, sr)
        print(f"📄 Transcribing {len(chunks)} chunks...")
        return "\n\n".join([self._transcribe_chunk(c) for c in chunks])

class BatchProcessor:
    def __init__(self, input_dir, audio_dir, output_dir):
        self.input_dir = input_dir
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.converter = VideoConverter()
        self.transcriber = WhisperTranscriber()

        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # 🔁 Full transcription pipeline
    def process_all(self):
        mp4_files = [f for f in os.listdir(self.input_dir) if f.endswith(".mp4")]
        for fname in mp4_files:
            print(f"\n🎬 Processing {fname}")
            base = os.path.splitext(fname)[0]
            mp4_path = os.path.join(self.input_dir, fname)
            wav_path = os.path.join(self.audio_dir, f"{base}.wav")
            txt_path = os.path.join(self.output_dir, f"{base}.txt")

            self.converter.convert(mp4_path, wav_path)
            text = self.transcriber.transcribe(wav_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved transcript: {txt_path}")

# 🏁 Run
if __name__ == "__main__":
    processor = BatchProcessor(
        input_dir="mp4_videos",
        audio_dir="wav_audio",
        output_dir="transcripts"
    )
    processor.process_all()
