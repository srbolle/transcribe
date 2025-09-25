#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local transcription tool using faster-whisper (GPU if available) with optional
speaker diarization. Two options:
 - No-HF: Resemblyzer + sklearn clustering (default if no HF token)
 - pyannote.audio (requires Hugging Face token)

Usage examples:
  python transcribe.py --input /home/user/soundfiles/Vilde_20250919.mp3 \
    --model large-v3 --out-formats txt srt --diarize --language auto

Notes:
 - GPU is used automatically if available by faster-whisper.
 - SRT output will include speaker labels like Speaker 1, Speaker 2.
 - For best diarization without Hugging Face, provide --num-speakers.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    print("Missing dependency: faster-whisper. Install requirements first.", file=sys.stderr)
    raise

try:
    import srt as srt_lib
except ImportError:
    srt_lib = None

# Optional imports for diarization
try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:
    PyannotePipeline = None

# No-HF diarization (Resemblyzer + sklearn)
try:
    from resemblyzer import VoiceEncoder
    import librosa
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    VoiceEncoder = None
    librosa = None
    AgglomerativeClustering = None


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def write_txt(path: str, utterances: List[Utterance]):
    with open(path, "w", encoding="utf-8") as f:
        for utt in utterances:
            spk = f"{utt.speaker}: " if utt.speaker else ""
            f.write(f"[{format_timestamp(utt.start)} - {format_timestamp(utt.end)}] {spk}{utt.text}\n")


def write_srt(path: str, utterances: List[Utterance]):
    if srt_lib is None:
        # Fallback rudimentary writer
        with open(path, "w", encoding="utf-8") as f:
            for i, utt in enumerate(utterances, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(utt.start)} --> {format_timestamp(utt.end)}\n")
                if utt.speaker:
                    f.write(f"{utt.speaker}: ")
                f.write(utt.text.strip() + "\n\n")
        return

    import datetime as dt
    items = []
    for i, utt in enumerate(utterances, 1):
        content = f"{utt.speaker}: {utt.text}" if utt.speaker else utt.text
        start = dt.timedelta(seconds=float(max(0.0, utt.start)))
        end = dt.timedelta(seconds=float(max(utt.start + 0.2, utt.end)))
        items.append(srt_lib.Subtitle(index=i, start=start, end=end, content=content))
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt_lib.compose(items))


def load_diarization_pipeline(hf_token: Optional[str]):
    if PyannotePipeline is None:
        print("pyannote.audio not installed. Install it to enable diarization.", file=sys.stderr)
        return None
    if not hf_token:
        print("HUGGINGFACE_TOKEN is not set. Diarization models require a Hugging Face token.", file=sys.stderr)
        return None
    try:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        return pipeline
    except Exception as e:
        print(f"Failed to load pyannote pipeline: {e}", file=sys.stderr)
        return None


def apply_diarization(audio_path: str, num_speakers: Optional[int], hf_token: Optional[str]):
    # Try HF-based first if token provided; else None
    if hf_token:
        pipeline = load_diarization_pipeline(hf_token)
        if pipeline is not None:
            try:
                diarization = pipeline(audio_path, num_speakers=num_speakers)
                segments = []
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append((float(segment.start), float(segment.end), str(speaker)))
                segments.sort(key=lambda x: x[0])
                return segments
            except Exception as e:
                print(f"Diarization (pyannote) failed: {e}", file=sys.stderr)
                # fallthrough to no-HF path
    return None


def diarize_by_embeddings(audio_path: str, utterances: List[Utterance], num_speakers: Optional[int]) -> Optional[List[str]]:
    """No-HF diarization: cluster utterance-level embeddings into speakers.
    Returns a list of speaker labels aligned with utterances order (e.g., S1, S2...).
    """
    if VoiceEncoder is None or librosa is None or AgglomerativeClustering is None:
        print("Resemblyzer/librosa/sklearn not installed; cannot perform no-HF diarization.", file=sys.stderr)
        return None
    if not utterances:
        return []
    try:
        # Load audio once
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Force CPU to avoid cuDNN / CUDA requirements for diarization
        try:
            encoder = VoiceEncoder(device="cpu")
        except TypeError:
            # Older resemblyzer versions may not support the device kwarg explicitly
            encoder = VoiceEncoder()
        # Compute embedding per utterance using its time span
        embs = []
        valid_idx = []
        for i, utt in enumerate(utterances):
            s = max(0, int(utt.start * 16000))
            e = min(len(wav), int(utt.end * 16000))
            if e - s < 1600:  # <0.1s skip (too short)
                embs.append(np.zeros(256, dtype=np.float32))
                valid_idx.append(False)
                continue
            seg = wav[s:e]
            emb = encoder.embed_utterance(seg)
            embs.append(emb.astype(np.float32))
            valid_idx.append(True)
        X = np.vstack(embs)
        # If num_speakers unknown, choose 2-6 by silhouette; here we require user input for stability
        n_spk = num_speakers or 2
        clust = AgglomerativeClustering(n_clusters=n_spk)
        labels = clust.fit_predict(X)
        # Map to Speaker 1..n order of appearance
        seen = {}
        next_id = 1
        spk_labels: List[str] = []
        for lab in labels:
            if lab not in seen:
                seen[lab] = f"Speaker {next_id}"
                next_id += 1
            spk_labels.append(seen[lab])
        return spk_labels
    except Exception as e:
        print(f"No-HF diarization failed: {e}", file=sys.stderr)
        return None


def assign_speakers(utterances: List[Utterance], diar_segments: List[Tuple[float, float, str]]):
    # Greedy overlap assignment: assign each utterance to the speaker with max overlap
    for utt in utterances:
        best_speaker = None
        best_overlap = 0.0
        for ds, de, spk in diar_segments:
            # compute overlap
            start = max(utt.start, ds)
            end = min(utt.end, de)
            overlap = max(0.0, end - start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk
        if best_speaker is not None:
            utt.speaker = best_speaker


def merge_adjacent(utterances: List[Utterance], max_gap: float = 0.5, max_len: int = 250) -> List[Utterance]:
    if not utterances:
        return []
    merged: List[Utterance] = []
    cur = Utterance(**utterances[0].__dict__)
    for nxt in utterances[1:]:
        same_spk = (cur.speaker == nxt.speaker)
        gap = max(0.0, nxt.start - cur.end)
        if same_spk and gap <= max_gap and len(cur.text) + 1 + len(nxt.text) <= max_len:
            cur.end = max(cur.end, nxt.end)
            cur.text = (cur.text + " " + nxt.text).strip()
        else:
            merged.append(cur)
            cur = Utterance(**nxt.__dict__)
    merged.append(cur)
    return merged


def transcribe(
    input_path: str,
    model_size: str = "large-v3",
    language: Optional[str] = None,
    beam_size: int = 5,
    compute_type: str = "auto",
    device: str = "auto",
    vad_filter: bool = True,
    diarize: bool = False,
    num_speakers: Optional[int] = None,
) -> List[Utterance]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Initialize WhisperModel (auto-detect GPU) with safe fallback to CPU
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Whisper initialization failed with device='{device}' ({e}). Falling back to CPU...", file=sys.stderr)
        model = WhisperModel(model_size, device="cpu", compute_type="int8" if compute_type == "auto" else compute_type)

    # Transcription
    segments_gen, info = model.transcribe(
        input_path,
        language=None if language in (None, "auto", "AUTO") else language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=False,
    )

    utterances: List[Utterance] = []
    for seg in segments_gen:
        utterances.append(Utterance(start=float(seg.start or 0.0), end=float(seg.end or 0.0), text=seg.text.strip()))

    # Diarization
    if diarize:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        used_any = False
        # Try HF-based first if token present
        if hf_token:
            diar_segments = apply_diarization(input_path, num_speakers=num_speakers, hf_token=hf_token)
            if diar_segments:
                assign_speakers(utterances, diar_segments)
                utterances = merge_adjacent(utterances)
                used_any = True
        # If not used, do no-HF diarization by clustering utterance embeddings
        if not used_any:
            spk_labels = diarize_by_embeddings(input_path, utterances, num_speakers=num_speakers)
            if spk_labels:
                for u, lbl in zip(utterances, spk_labels):
                    u.speaker = lbl
                utterances = merge_adjacent(utterances)
                used_any = True
        if not used_any:
            print("Warning: Diarization requested but could not be performed. Proceeding without speaker labels.", file=sys.stderr)

    return utterances


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio locally with optional diarization.")
    parser.add_argument("--input", required=True, help="Path to input audio/video file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", default="auto", help="Language code or 'auto'")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--compute-type", default="auto", help="Compute type: auto, float16, float32, int8, etc.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (no-HF by default; pyannote if HUGGINGFACE_TOKEN set)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Known number of speakers (optional)")
    parser.add_argument("--out-formats", nargs="+", default=["txt", "srt"], help="Output formats: txt srt vtt")
    parser.add_argument("--out-dir", default="outputs", help="Directory to write outputs")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    utterances = transcribe(
        input_path=input_path,
        model_size=args.model,
        language=args.language,
        beam_size=args.beam_size,
        compute_type=args.compute_type,
        device=args.device,
        vad_filter=not args.no_vad,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
    )

    base = os.path.splitext(os.path.basename(input_path))[0]
    if "txt" in args.out_formats:
        write_txt(os.path.join(args.out_dir, f"{base}.txt"), utterances)
    if "srt" in args.out_formats:
        write_srt(os.path.join(args.out_dir, f"{base}.srt"), utterances)
    if "vtt" in args.out_formats:
        # Simple VTT writer
        vtt_path = os.path.join(args.out_dir, f"{base}.vtt")
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for utt in utterances:
                start = format_timestamp(utt.start).replace(",", ".")
                end = format_timestamp(utt.end).replace(",", ".")
                line = f"{utt.speaker}: {utt.text}" if utt.speaker else utt.text
                f.write(f"{start} --> {end}\n{line}\n\n")

    print(f"Done. Outputs saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
