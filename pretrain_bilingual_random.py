import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def main():
    # Setup Tokenizer (Using the joint tokenizer to match Model A and Model C)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_joint_tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # Setup Model Architecture
    model_id = "BabyLM-community/babylm-baseline-10m-gpt2"
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    
    print("Initializing architecture framework with initialized random weights...")
    model = AutoModelForCausalLM.from_config(config)

    # Setup Dataset
    print("Loading full bilingual dataset...")
    full_dataset = load_dataset("json", data_files="data/sorted_bilingual_training_data.jsonl", split="train")

    # Tokenization subroutine (Extracting natural language text ONLY, skipping POS)
    def tokenize_text(examples):
        return tokenizer(examples["text_block"], truncation=True, max_length=256, padding="max_length")

    print("Tokenizing dataset (text_block only)...")
    tokenized_dataset = full_dataset.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)

    # Standard random shuffling for Vanilla training
    print("Shuffling dataset randomly for Ablation Study...")
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # TRAINING: RANDOM BILINGUAL IMMERSION (10 Epochs)
    # =========================================================================
    # Training for 10 epochs to perfectly match Model A (English) and Model C (Curriculum)
    print("\n=== RUNNING BASELINE: RANDOM BILINGUAL IMMERSION ===")
    
    training_args = TrainingArguments(
        output_dir="./babylm_bilingual_random_output",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model("./babylm_bilingual_random_output")
    print("Random Bilingual training completed.")

if __name__ == "__main__":
    main()