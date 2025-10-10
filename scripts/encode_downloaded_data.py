#!/usr/bin/env python3
"""
Encode already-downloaded Vietnamese dataset with NeuCodec.
Converts audio to VQ codes for TTS training.
"""

import os
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'

from datasets import load_dataset, Audio
import json
from pathlib import Path
from tqdm import tqdm
import torch
from neucodec import NeuCodec
import numpy as np
import soundfile as sf
import io


def load_cached_dataset(dataset_name):
    """Load dataset from HuggingFace cache (no re-download)."""
    print(f"📂 Loading {dataset_name} from cache...")

    if dataset_name == "infore1":
        dataset = load_dataset("doof-ferb/infore1_25hours", split="train")
    elif dataset_name == "common_voice":
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "vi", split="train[:10000]")
    elif dataset_name == "vivos":
        dataset = load_dataset("AILAB-VNUHCM/vivos", split="train")
    elif dataset_name == "vlsp2020":
        dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Prevent automatic audio decoding
    dataset = dataset.cast_column("audio", Audio(decode=False))

    print(f"✅ Loaded {len(dataset)} samples from cache")
    return dataset


def get_text_from_sample(sample, dataset_name):
    """Extract text from sample based on dataset."""
    # Each dataset has its own text field name
    field_map = {
        'infore1': 'transcription',
        'common_voice': 'sentence',
        'vivos': 'text',
        'vlsp2020': 'transcription'
    }
    field_name = field_map.get(dataset_name, 'text')
    return sample.get(field_name, '')


def decode_audio(audio_dict):
    """Decode audio from various formats."""
    # Priority 1: bytes (raw audio data)
    if 'bytes' in audio_dict and audio_dict['bytes']:
        audio_array, sr = sf.read(io.BytesIO(audio_dict['bytes']))
        return audio_array, sr

    # Priority 2: path to file
    if 'path' in audio_dict and audio_dict['path']:
        audio_array, sr = sf.read(audio_dict['path'])
        return audio_array, sr

    # Priority 3: already decoded array
    if 'array' in audio_dict:
        audio_array = audio_dict['array']
        sr = audio_dict.get('sampling_rate', 16000)
        return audio_array, sr

    return None, None


def resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio_array

    duration = len(audio_array) / orig_sr
    new_length = int(duration * target_sr)
    return np.interp(
        np.linspace(0, len(audio_array), new_length),
        np.arange(len(audio_array)),
        audio_array
    )


def encode_dataset(dataset_name="infore1", output_dir="data/vietnamese_encoded", num_samples=None, batch_size=32):
    """
    Encode cached dataset with NeuCodec.

    Args:
        dataset_name: Dataset to encode (infore1, common_voice, vivos, vlsp2020)
        output_dir: Output directory for encoded data
        num_samples: Limit number of samples to encode (None = all)
        batch_size: Batch size for codec encoding (default: 32)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎵 Vietnamese TTS Data Encoder (Batch Mode)")
    print("=" * 60)
    print()

    # Load dataset
    dataset = load_cached_dataset(dataset_name)

    # Limit samples if requested
    if num_samples and num_samples < len(dataset):
        print(f"⚡ Limiting to first {num_samples:,} samples for quick test")
        dataset = dataset.select(range(num_samples))
        print(f"✅ Selected {len(dataset)} samples")

    # Load NeuCodec
    print("\n🔧 Loading NeuCodec...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")
    print("✅ Codec loaded")

    # Encode samples in batches
    print(f"\n🔄 Encoding {len(dataset)} samples in batches of {batch_size}...")
    encoded_samples = []
    failed = 0

    # Collect samples in batches
    batch_texts = []
    batch_audios = []
    batch_indices = []

    for idx, sample in enumerate(tqdm(dataset)):
        try:
            # Get text
            text = get_text_from_sample(sample, dataset_name)
            if not text or len(text.strip()) < 5:
                continue

            # Get and decode audio
            audio_dict = sample['audio']
            if not isinstance(audio_dict, dict):
                continue

            audio_array, sr = decode_audio(audio_dict)
            if audio_array is None:
                continue

            # Resample to 16kHz if needed
            audio_array = resample_audio(audio_array, sr, 16000)

            # Convert to float32
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            else:
                audio_array = audio_array.astype(np.float32)

            # Add to batch
            batch_texts.append(text.strip())
            batch_audios.append(audio_array)
            batch_indices.append(idx)

            # Process batch when full
            if len(batch_audios) >= batch_size:
                # Encode batch with NeuCodec
                with torch.no_grad():
                    for i, audio in enumerate(batch_audios):
                        wav_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
                        codes = codec.encode_code(wav_tensor).squeeze().cpu().numpy().tolist()

                        encoded_samples.append({
                            "text": batch_texts[i],
                            "codes": codes,
                            "__key__": f"{dataset_name}_{batch_indices[i]:06d}"
                        })

                # Clear batch
                batch_texts = []
                batch_audios = []
                batch_indices = []

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n⚠️  Error on sample {idx}: {e}")
            continue

    # Process remaining samples in batch
    if batch_audios:
        print(f"\n🔄 Processing final batch of {len(batch_audios)} samples...")
        with torch.no_grad():
            for i, audio in enumerate(batch_audios):
                wav_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
                codes = codec.encode_code(wav_tensor).squeeze().cpu().numpy().tolist()

                encoded_samples.append({
                    "text": batch_texts[i],
                    "codes": codes,
                    "__key__": f"{dataset_name}_{batch_indices[i]:06d}"
                })

    # Save encoded data
    if num_samples:
        output_file = output_dir / f"{dataset_name}_{num_samples}_encoded.json"
    else:
        output_file = output_dir / f"{dataset_name}_encoded.json"
    print(f"\n💾 Saving encoded data...")

    # Write to temp file first to avoid corruption if interrupted
    temp_file = output_dir / f"{output_file.name}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(encoded_samples, f, ensure_ascii=False, indent=2)

    # Atomic rename - only happens if write was successful
    temp_file.rename(output_file)
    print(f"✅ Saved to {output_file}")

    # Statistics
    print("\n" + "=" * 60)
    print("✅ ENCODING COMPLETE!")
    print("=" * 60)
    print(f"   Successfully encoded: {len(encoded_samples):,} samples")
    print(f"   Failed: {failed:,} samples")
    print(f"   Output file: {output_file}")

    if len(encoded_samples) > 0:
        total_codes = sum(len(s['codes']) for s in encoded_samples)
        avg_codes = total_codes / len(encoded_samples)
        estimated_hours = (total_codes / 50) / 3600

        print(f"\n📊 Dataset Statistics:")
        print(f"   Samples: {len(encoded_samples):,}")
        print(f"   Avg codes per sample: {avg_codes:.1f}")
        print(f"   Estimated duration: {estimated_hours:.2f} hours")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode Vietnamese dataset with NeuCodec")
    parser.add_argument(
        "--dataset",
        type=str,
        default="infore1",
        choices=["infore1", "common_voice", "vivos", "vlsp2020"],
        help="Which dataset to encode"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vietnamese_encoded",
        help="Output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples to encode (default: all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )

    args = parser.parse_args()

    try:
        output_file = encode_dataset(
            args.dataset,
            args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )
        print(f"\n🎉 Done! Encoded data saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
