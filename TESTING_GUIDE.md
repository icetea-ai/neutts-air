# Vietnamese Finetune Script - Testing Guide

## Current Status

✅ **Python Syntax**: Valid (no syntax errors)
✅ **Dependencies Listed**: All required packages in `requirements.txt`
✅ **Code Structure**: Follows proper patterns
⚠️ **Runtime Validation**: Requires environment setup to test

---

## Testing Without a Dev Environment

Since you're running this through the web interface without a full Python environment, here are alternative ways to validate the script:

### Option 1: Use Google Colab (Recommended - FREE)

1. **Go to**: https://colab.research.google.com
2. **Create a new notebook**
3. **Run these commands**:

```python
# Cell 1: Clone the repository
!git clone https://github.com/icetea-ai/neutts-air.git
%cd neutts-air
!git checkout feat/vn-finetunning

# Cell 2: Install dependencies
!pip install -q torch transformers datasets omegaconf fire loguru

# Cell 3: Run validation test
!python test_vietnamese_finetune.py
```

This will test:
- ✓ All imports work
- ✓ Config loads correctly
- ✓ Dataset is accessible
- ✓ Tokenizer loads
- ✓ Data preprocessing works

**Time**: ~5-10 minutes
**Cost**: FREE

---

### Option 2: Use Kaggle Notebooks (FREE GPU)

1. **Go to**: https://www.kaggle.com/code
2. **Create new notebook**
3. **Enable GPU**: Settings → Accelerator → GPU
4. **Run the same commands as Colab**

---

### Option 3: Quick GitHub Codespaces (FREE tier available)

1. **Go to**: https://github.com/icetea-ai/neutts-air
2. **Click**: Code → Codespaces → Create codespace
3. **In terminal**:
```bash
git checkout feat/vn-finetunning
pip install -r requirements.txt
python test_vietnamese_finetune.py
```

---

### Option 4: Minimal Validation (No Environment Needed)

I can perform these checks right now:

#### ✅ Code Review Checklist

| Check | Status | Notes |
|-------|--------|-------|
| Python syntax valid | ✅ PASS | No syntax errors |
| All imports in requirements.txt | ✅ PASS | torch, transformers, datasets, etc. |
| Config file exists | ✅ PASS | `configs/vietnamese_finetune.yaml` |
| Script follows HF Trainer pattern | ✅ PASS | Standard Trainer API usage |
| Data preprocessing defined | ✅ PASS | `preprocess_sample()` function |
| Special tokens handled | ✅ PASS | `<|SPEECH_GENERATION_START|>` |
| Training args configured | ✅ PASS | bf16, learning rate, etc. |

#### ⚠️ Issues Found (from code review):

1. **Line 91**: Uses `dtype="auto"` instead of `torch_dtype="auto"`
2. **Line 103**: Hardcoded sample limit `train[:2000]`
3. **Line 122**: Very high worker count (64) may cause issues
4. **Config mismatch**: Config mentions unused parameters

---

## What the Script Will Do (Execution Flow)

```
1. Load config from configs/vietnamese_finetune.yaml
   └─ restore_from: "neuphonic/neutts-air"
   └─ max_steps: 70000
   └─ batch_size: 8

2. Download/load model and tokenizer
   └─ Model: neuphonic/neutts-air (from HuggingFace)
   └─ Size: ~1-2GB download

3. Load Vietnamese dataset
   └─ Dataset: pnnbao-ump/VieNeuCodec-dataset
   └─ Samples: 2000 (currently hardcoded)
   └─ Features: phones (text) + codes (audio tokens)

4. Preprocess data
   └─ Filter empty phones
   └─ Create chat format: "user: Convert text... assistant: <speech>"
   └─ Tokenize to max_len=2048
   └─ Create attention masks and labels

5. Train model
   └─ Using HuggingFace Trainer
   └─ bf16 precision
   └─ Save checkpoints every 8750 steps

6. Save final model
   └─ Location: ./checkpoints/vietnamese-tts-full/
```

---

## Expected Behavior

### On First Run:
```
Loading config from configs/vietnamese_finetune.yaml
Logging to: ./checkpoints/vietnamese-tts-full
Loading checkpoint from neuphonic/neutts-air
[Downloads model: ~1-2GB, takes 2-5 minutes]
[Downloads dataset: ~500MB, takes 1-3 minutes]
[Filters dataset and preprocesses]
[Training starts with progress bar]
```

### During Training:
- Progress bar with loss metrics
- Saves checkpoint every 8750 steps
- Logs every 500 steps
- Uses bf16 (requires GPU with bf16 support)

### After Completion:
- Final model saved to `./checkpoints/vietnamese-tts-full/`
- Can be loaded with: `AutoModelForCausalLM.from_pretrained("./checkpoints/vietnamese-tts-full/")`

---

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 16GB VRAM | 24GB+ VRAM |
| RAM | 16GB | 32GB+ |
| Disk | 10GB | 20GB+ |
| Time | 2-4 hours | 1-2 hours (with better GPU) |

---

## Quick Syntax-Only Validation (Already Done ✅)

```bash
# This already passed:
python -m py_compile examples/finetune_vietnamese.py
# No errors = valid Python syntax
```

---

## How to Test Properly

**Best approach if you want to verify it works**:

1. **Open Google Colab** (free, no setup needed)
2. **Paste this one-liner**:

```python
!git clone https://github.com/icetea-ai/neutts-air.git && \
cd neutts-air && \
git checkout feat/vn-finetunning && \
pip install -q -r requirements.txt && \
python test_vietnamese_finetune.py
```

3. **Wait 5-10 minutes** for the test to complete
4. **Check output** - should see "🎉 All tests passed!"

---

## Summary

**Without dependencies installed**, I can confirm:
- ✅ Code is syntactically correct
- ✅ All imports exist in requirements.txt
- ✅ Follows proper patterns for HuggingFace training
- ⚠️ Has minor issues (see review above) but should work

**To fully test**, use **Google Colab** (easiest, free, 5 minutes).

**Ready to run?** Just fix the minor issues first (or proceed as-is for testing).

Would you like me to fix the issues before you test?
