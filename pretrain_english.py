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
    # Load the EXACT same bilingual tokenizer to guarantee identical parameter count
    # for a fair A/B test between the English baseline and the Bilingual model.
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_joint_tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # Setup Model Architecture
    model_id = "BabyLM-community/babylm-baseline-10m-gpt2"
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    
    print("Initializing architecture framework with initialized random weights...")
    model = AutoModelForCausalLM.from_config(config)

    # Load Dataset
    print("Loading English-only dataset...")
    dataset = load_dataset("json", data_files="data/english_only_training_data.jsonl", split="train")

    # Tokenization subroutine
    def tokenize_text(examples):
        # We assume the English file still uses the 'text' key
        texts_with_eos = [text + " <|endoftext|>" for text in examples["text"]]
        return tokenizer(texts_with_eos)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_text, batched=True, remove_columns=dataset.column_names)
    
    block_size = 256

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder to ensure exact block sizes
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    print("Packing sequences to optimize GPU compute...")
    packed_dataset = tokenized_dataset.map(group_texts, batched=True)

    # Standard random shuffling for Vanilla training
    packed_dataset = packed_dataset.shuffle(seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # BASELINE RUN: FULL ENGLISH IMMERSION (10 Epochs)
    # =========================================================================
    # We run 10 epochs here to perfectly match the total compute time 
    # of the bilingual model (1 Phase 1 epoch + 9 Phase 2 epochs).
    print("\n=== RUNNING BASELINE: ENGLISH ONLY IMMERSION ===")
    
    training_args = TrainingArguments(
        output_dir="./babylm_baseline_output",
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
        train_dataset=packed_dataset,
    )

    trainer.train()
    trainer.save_model("./babylm_pretrain_english_baseline")
    print("Baseline English training completed.")

if __name__ == "__main__":
    main()