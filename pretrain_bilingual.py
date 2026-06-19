import os
import torch
from torch.utils.data import SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets

# Custom Trainer to disable automatic shuffling for Curriculum Learning
class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        # Enforce strict sequential order to keep Phase 1 structure (POS -> Text)
        target_dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(target_dataset)
    
def main():
    # Setup Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_bilingual_tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # Setup Model Architecture
    model_id = "BabyLM-community/babylm-baseline-10m-gpt2"
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    
    print("Initializing architecture framework with initialized random weights...")
    model = AutoModelForCausalLM.from_config(config)

    # Setup Dataset
    full_dataset = load_dataset("json", data_files="data/sorted_bilingual_training_data.jsonl", split="train")

    # Separate raw datasets strictly using pre-defined pipeline tags
    print("Filtering datasets into training partitions...")
    stage1_raw = full_dataset.filter(lambda x: x["stage"] == 1)
    stage2_raw = full_dataset.filter(lambda x: x["stage"] == 2)

    # Tokenization subroutines
    def tokenize_pos(examples):
        return tokenizer(examples["pos_block"], truncation=True, max_length=256, padding="max_length")

    def tokenize_text(examples):
        return tokenizer(examples["text_block"], truncation=True, max_length=256, padding="max_length")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # PHASE 1: SCAFFOLDING BOOTSTRAP (1 Epoch)
    # =========================================================================
    print("\n=== RUNNING PHASE 1: BALANCED STRUCTURAL SCAFFOLDING (POS) ===")
    
    # Tokenize the 1M word subset using the pure POS text stream
    tokenized_stage1_pos = stage1_raw.map(tokenize_pos, batched=True, remove_columns=full_dataset.column_names)
    tokenized_stage2_text = stage2_raw.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)

    phase1_dataset = concatenate_datasets([tokenized_stage1_pos, tokenized_stage2_text])

    phase1_args = TrainingArguments(
        output_dir="./babylm_phase1_output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer_phase1 = CurriculumTrainer(
        model=model,
        args=phase1_args,
        data_collator=data_collator,
        train_dataset=phase1_dataset,
    )
    
    trainer_phase1.train()
    trainer_phase1.save_model("./babylm_phase1_bilingual")
    print("Phase 1 completed.")

    # =========================================================================
    # PHASE 2: LEXICAL IMMERSION (9 Epochs)
    # =========================================================================
    print("\n=== RUNNING PHASE 2: FULL LEXICAL IMMERSION (TEXT) ===")
    
    # Load model from the previous structural bootstrap checkpoint
    model = AutoModelForCausalLM.from_pretrained("./babylm_phase1_bilingual")
    
    # Tokenize the 9M word subset using the actual natural language text stream
    phase2_dataset = full_dataset.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)
    
    # Shuffle phase 2 sequence to maximize optimization robustness across epochs
    phase2_dataset = phase2_dataset.shuffle(seed=42)

    phase2_args = TrainingArguments(
        output_dir="./babylm_phase2_output",
        num_train_epochs=9,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer_phase2 = Trainer(
        model=model,
        args=phase2_args,
        data_collator=data_collator,
        train_dataset=phase2_dataset,
    )

    trainer_phase2.train()
    trainer_phase2.save_model("./babylm_pretrain_bilingual")
    print("Phase 2 completed.")

if __name__ == "__main__":
    main()