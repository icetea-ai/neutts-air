#!/usr/bin/env python3
"""
Vietnamese TTS Finetuning Script for NeuTTS Air.
Adapted from the original English finetuning script.
"""

import os
import warnings
import re
import torch

# Fix espeak library path (must be before importing phonemizer)
import platform
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    system = platform.system()

    if system == "Darwin":  # macOS
        # Try common Homebrew espeak library locations
        possible_paths = [
            '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib',  # Homebrew Apple Silicon
            '/usr/local/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib',      # Homebrew Intel
            '/opt/homebrew/lib/libespeak.dylib',
            '/usr/local/lib/libespeak.dylib',
        ]

        for path in possible_paths:
            if os.path.exists(path):
                EspeakWrapper.set_library(path)
                print(f"✅ Set espeak library: {path}")
                break
    elif system == "Linux":  # Ubuntu/Linux
        # Try common Linux espeak library locations
        possible_paths = [
            '/usr/lib/x86_64-linux-gnu/libespeak.so.1',  # Ubuntu/Debian x64
            '/usr/lib/aarch64-linux-gnu/libespeak.so.1',  # Ubuntu/Debian ARM64
            '/usr/lib/libespeak.so.1',
            '/usr/lib/libespeak.so',
            '/usr/local/lib/libespeak.so',
        ]

        for path in possible_paths:
            if os.path.exists(path):
                EspeakWrapper.set_library(path)
                print(f"✅ Set espeak library: {path}")
                break
    # For other systems, let phonemizer auto-detect

except Exception as e:
    print(f"⚠️  Could not set espeak library (will try auto-detect): {e}")

import phonemizer
from viphoneme import vi2IPA_split

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from loguru import logger as LOGGER
from datasets import load_dataset
import json

warnings.filterwarnings("ignore")


def vietnamese_data_filter(sample):
    """Filter for Vietnamese text samples."""
    text = sample["text"]

    # Basic length check (allow very short for testing)
    if len(text) < 2 or len(text) > 500:
        return False

    # Just check it's not empty or whitespace only
    if not text.strip():
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):
    """Preprocess Vietnamese sample for training."""

    # Get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100

    # Unpack sample
    vq_codes = sample["codes"]
    text = sample["text"]

    # Phonemize Vietnamese text
    phones = g2p.phonemize([text])

    # Safety check
    if not phones or not phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for sample: {sample['__key__']} text={text}")
        return None

    phones = phones[0].split()
    phones = ' '.join(phones)

    # Convert codes to string format
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # Create chat format
    chat = f"""<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    ids = tokenizer.encode(chat)

    # Pad or truncate to max_len
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # Convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    # Create labels (only train on speech generation part)
    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Return in HuggingFace format
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str, device: str = "auto"):
    """Main training function.

    Args:
        config_fpath: Path to config YAML file
        device: Device to use - "auto" (default), "cpu", "cuda", or "mps"
    """

    # Setup device
    if device == "cpu":
        # Force CPU training
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_PREFER_CPU'] = '1'
        if torch.backends.mps.is_available():
            torch.backends.mps.is_available = lambda: False
        torch.set_default_device('cpu')
        use_gpu = False
        print("⚠️  MPS disabled - training on CPU")
    elif device == "auto":
        # Auto-detect best device
        if torch.cuda.is_available():
            use_gpu = True
            device = "cuda"
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            use_gpu = True
            device = "mps"
            print("✅ Using Apple Silicon GPU (MPS)")
        else:
            use_gpu = False
            device = "cpu"
            print("⚠️  No GPU available - using CPU")
    else:
        # User specified device
        use_gpu = device in ["cuda", "mps"]
        print(f"✅ Using device: {device}")

    # Load config
    print("\n" + "=" * 60)
    print("🇻🇳 Vietnamese TTS Finetuning")
    print("=" * 60)
    print(f"\n📄 Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)

    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    LOGGER.info(f"💾 Checkpoints will be saved to: {checkpoints_dir}")

    restore_from = config.restore_from

    # Load model and tokenizer
    print(f"\n🔧 Loading model from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)

    # Configure model loading based on device
    if use_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            restore_from,
            torch_dtype="auto",
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            restore_from,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    print(f"📍 Model device: {model.device}")

    # Initialize Vietnamese phonemizer (viphoneme)
    print(f"\n🗣️  Initializing Vietnamese phonemizer")
    print("✅ Using viphoneme (Vietnamese-specific G2P with accurate tone handling)")

    class ViphonemeWrapper:
        """Wrapper to make viphoneme compatible with phonemizer API."""
        def phonemize(self, texts, strip=True):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for text in texts:
                # Convert to IPA with space-separated phonemes
                phones = vi2IPA_split(text, delim=' ')
                results.append(phones)
            return results

    g2p = ViphonemeWrapper()

    # Create preprocessing function
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    # Load Vietnamese dataset from JSON
    data_file = config.data_file
    print(f"\n📂 Loading Vietnamese dataset from {data_file}")

    # Load JSON manually to avoid Arrow parsing issues
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    from datasets import Dataset
    dataset = Dataset.from_list(data)
    print(f"✅ Loaded {len(dataset)} samples")

    # Optional: Create validation split
    if config.get('validation_split', 0) > 0:
        val_size = config.validation_split
        print(f"\n🔀 Creating validation split ({val_size*100:.1f}%)")
        dataset = dataset.train_test_split(test_size=val_size, seed=config.get('seed', 42))
        train_dataset = dataset['train']
        val_dataset = dataset['test']
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val:   {len(val_dataset)} samples")
    else:
        train_dataset = dataset
        val_dataset = None

    # Filter and preprocess
    print("\n🔄 Filtering and preprocessing data...")
    train_dataset = train_dataset.filter(vietnamese_data_filter)
    train_dataset = train_dataset.map(partial_preprocess, remove_columns=["text", "codes"])

    # Remove None samples (failed preprocessing)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    print(f"✅ Processed {len(train_dataset)} training samples")

    if val_dataset:
        val_dataset = val_dataset.filter(vietnamese_data_filter)
        val_dataset = val_dataset.map(partial_preprocess, remove_columns=["text", "codes"])
        val_dataset = val_dataset.filter(lambda x: x is not None)
        print(f"✅ Processed {len(val_dataset)} validation samples")

    # Training arguments
    print("\n⚙️  Setting up training configuration...")
    has_validation = val_dataset is not None and len(val_dataset) > 0

    # Enable mixed precision for GPU training
    use_bf16 = use_gpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = use_gpu and not use_bf16

    if use_bf16:
        print("   Using BF16 mixed precision")
    elif use_fp16:
        print("   Using FP16 mixed precision")
    else:
        print("   Using FP32 (full precision)")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        do_eval=has_validation,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=config.get('eval_steps', config.save_steps) if has_validation else None,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_total_limit=3,  # Keep only last 3 checkpoints
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=False,  # Disable for compatibility
        dataloader_num_workers=4,  # Reduced for stability
        seed=config.get('seed', 42),
    )

    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # Train!
    print("\n" + "=" * 60)
    print("🎓 Starting Training")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Model saved to: {checkpoints_dir}")
    print(f"\n📝 To use your Vietnamese model:")
    print(f"""
    from neuttsair.neutts import NeuTTSAir

    tts = NeuTTSAir(
        backbone_repo="{checkpoints_dir}",
        language="{language}"
    )
    """)


if __name__ == "__main__":
    Fire(main)
