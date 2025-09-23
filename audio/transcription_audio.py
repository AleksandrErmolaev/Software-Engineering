# asr_whisper.py
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if len(audio.shape) > 1:  # стерео → моно
        audio = audio[:, 0]
    return audio, target_sr


def split_audio(audio, sr, chunk_length_s=30, stride_s=5):
    """
    Разбиваем аудио на куски с перекрытием.
    chunk_length_s = длина куска (сек)
    stride_s = перекрытие (сек)
    """
    chunk_len = chunk_length_s * sr
    stride = stride_s * sr
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_len, len(audio))
        chunk = audio[start:end]
        chunks.append(chunk)
        if end == len(audio):
            break
        start += chunk_len - stride
    return chunks


def transcribe_long_audio(path, model_name="openai/whisper-small", language="russian",
                          chunk_length_s=30, stride_s=5):
    device = get_device()
    print(f"Using device: {device}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    audio, sr = load_audio(path)
    chunks = split_audio(audio, sr, chunk_length_s, stride_s)

    texts = []
    for i, chunk in enumerate(chunks):
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        predicted_ids = model.generate(
            input_features,
            language=language,
            task="transcribe"
        )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        texts.append(transcription[0])
        print(f"[Chunk {i+1}/{len(chunks)}] {transcription[0]}")

    return " ".join(texts)


if __name__ == "__main__":
    audio_file = r"output3.wav"  # путь к файлу
    text = transcribe_long_audio(audio_file, model_name="openai/whisper-small", language="russian")
    print("---- Full Transcript ----")
    print(text)
