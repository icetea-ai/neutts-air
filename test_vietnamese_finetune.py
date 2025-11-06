#!/usr/bin/env python3
"""
Validation script for Vietnamese finetune script.
Tests the script without running full training.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test if all required imports work"""
    print("=" * 60)
    print("TEST 1: Checking Imports")
    print("=" * 60)
    try:
        import torch
        print(f"✓ torch {torch.__version__}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"✓ transformers")

        from datasets import load_dataset
        print(f"✓ datasets")

        from omegaconf import OmegaConf
        print(f"✓ omegaconf")

        from fire import Fire
        print(f"✓ fire")

        from loguru import logger
        print(f"✓ loguru")

        print("\n✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_config():
    """Test if config file loads correctly"""
    print("=" * 60)
    print("TEST 2: Loading Config File")
    print("=" * 60)
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/vietnamese_finetune.yaml")
        print(f"✓ Config loaded successfully")
        print(f"  - restore_from: {config.restore_from}")
        print(f"  - run_name: {config.run_name}")
        print(f"  - max_steps: {config.max_steps}")
        print(f"  - batch_size: {config.per_device_train_batch_size}")
        print(f"  - learning_rate: {config.lr}")
        print(f"  - max_seq_len: {config.max_seq_len}")
        print("\n✅ Config validation passed!\n")
        return True, config
    except Exception as e:
        print(f"\n❌ Config loading failed: {e}\n")
        return False, None


def test_dataset_access():
    """Test if dataset is accessible"""
    print("=" * 60)
    print("TEST 3: Checking Dataset Access")
    print("=" * 60)
    try:
        from datasets import load_dataset
        print("Attempting to load dataset: pnnbao-ump/VieNeuCodec-dataset")
        print("(loading only 5 samples for testing...)")

        dataset = load_dataset(
            "pnnbao-ump/VieNeuCodec-dataset",
            split="train[:5]",
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Features: {dataset.features}")

        # Check first sample
        sample = dataset[0]
        print(f"\n  Sample structure:")
        print(f"    - phones: {sample.get('phones', 'NOT FOUND')[:100] if sample.get('phones') else 'EMPTY'}...")
        print(f"    - codes length: {len(sample.get('codes', []))} tokens")

        print("\n✅ Dataset access successful!\n")
        return True, dataset
    except Exception as e:
        print(f"\n❌ Dataset access failed: {e}\n")
        print("This might be due to:")
        print("  - No internet connection")
        print("  - HuggingFace Hub unreachable")
        print("  - Dataset not available")
        return False, None


def test_tokenizer():
    """Test if tokenizer can be loaded"""
    print("=" * 60)
    print("TEST 4: Testing Tokenizer")
    print("=" * 60)
    try:
        from transformers import AutoTokenizer
        print("Loading tokenizer from: neuphonic/neutts-air")
        print("(this may take a moment on first run...)")

        tokenizer = AutoTokenizer.from_pretrained("neuphonic/neutts-air")

        print(f"✓ Tokenizer loaded successfully")
        print(f"  - Vocab size: {len(tokenizer)}")
        print(f"  - Pad token: {tokenizer.pad_token}")

        # Test special tokens
        test_tokens = ['<|SPEECH_GENERATION_START|>', '<|TEXT_PROMPT_START|>', '<|speech_0|>']
        print(f"\n  Special tokens check:")
        for token in test_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"    - {token}: ID={token_id}")

        print("\n✅ Tokenizer validation passed!\n")
        return True, tokenizer
    except Exception as e:
        print(f"\n❌ Tokenizer loading failed: {e}\n")
        return False, None


def test_preprocessing(dataset=None, tokenizer=None):
    """Test if data preprocessing works"""
    print("=" * 60)
    print("TEST 5: Testing Data Preprocessing")
    print("=" * 60)

    if dataset is None or tokenizer is None:
        print("⚠️  Skipping (dataset or tokenizer not available)\n")
        return False

    try:
        import torch
        from functools import partial

        # Import the preprocessing function from the script
        sys.path.insert(0, '/home/user/neutts-air/examples')
        from finetune_vietnamese import preprocess_sample, data_filter

        # Test filter
        print("Testing data filter...")
        sample = dataset[0]
        filter_result = data_filter(sample)
        print(f"  ✓ Filter result: {filter_result}")

        # Test preprocessing
        print("\nTesting preprocessing...")
        partial_preprocess = partial(
            preprocess_sample,
            tokenizer=tokenizer,
            max_len=2048,
        )

        processed = partial_preprocess(sample)

        if processed is None:
            print("  ⚠️  Preprocessing returned None (sample may have been filtered)")
        else:
            print(f"  ✓ Preprocessing successful")
            print(f"    - input_ids shape: {processed['input_ids'].shape}")
            print(f"    - labels shape: {processed['labels'].shape}")
            print(f"    - attention_mask shape: {processed['attention_mask'].shape}")
            print(f"    - Non-padding tokens: {processed['attention_mask'].sum().item()}")

        print("\n✅ Preprocessing validation passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Vietnamese Finetune Script Validation Test".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['config'], config = test_config()
    results['dataset'], dataset = test_dataset_access()
    results['tokenizer'], tokenizer = test_tokenizer()
    results['preprocessing'] = test_preprocessing(dataset, tokenizer)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The script should work correctly.")
        print("\nTo run the actual training:")
        print("  python examples/finetune_vietnamese.py configs/vietnamese_finetune.yaml")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("The script may not work correctly until these issues are resolved.")

    print("\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
