import warnings

import os
import torch

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from loguru import logger as LOGGER
from datasets import load_dataset


warnings.filterwarnings("ignore")

# Disable tokenizers parallelism to avoid warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def data_filter(sample):
    """Minimal filtering for Vietnamese phonemized text"""
    phones = sample["phones"]

    # Filter empty phones
    if len(phones) == 0:
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len):

    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100  # this is from LLaMA

    # unpack sample - phones are already provided!
    vq_codes = sample["codes"]
    phones = sample["phones"]

    # SAFE CHECK
    if not phones:
        LOGGER.warning(f"⚠️ Empty phones for sample")
        return None

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # get chat format
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    ids = tokenizer.encode(chat)

    # pad to make seq len
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # return in hf format
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):

    # load config
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")

    restore_from = config.restore_from

    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, dtype="auto")

    # No need for phonemizer - dataset already has phones!
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
    )

    # Load Vietnamese NeuCodec dataset
    vietnamese_dataset = load_dataset(
        "pnnbao-ump/VieNeuCodec-dataset",
        split="train[:2000]",
    )
    vietnamese_dataset = vietnamese_dataset.filter(data_filter).map(partial_preprocess, remove_columns=["phones", "codes"])

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
        dataloader_num_workers=64,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=vietnamese_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(checkpoints_dir)


if __name__ == "__main__":
    Fire(main)
