import os
import torch
import random
from torch.utils.data import SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets

# Custom Trainer to disable automatic shuffling for Curriculum Learning
class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        # Enforce strict sequential order to keep Phase 1 structure (Easy -> Hard POS)
        # For later epochs, the data is pre-shuffled internally before being concatenated
        target_dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(target_dataset)

def main():
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"

    # 1. Setup Official Tokenizer and Protect POS Tags
    print("Loading Official 2026 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Inject UPOS tags so BPE algorithm treats them as unbreakable single tokens
    upos_tags = ["NOUN", "PUNCT", "VERB", "PRON", "ADP", 
                "DET", "ADJ", "AUX", "ADV", "CCONJ", 
                "PROPN", "PART", "NUM", "SCONJ", "X", 
                "INTJ", "SYM"]
    tokenizer.add_special_tokens({'additional_special_tokens': upos_tags})
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # 2. Initialize Fresh Model
    print("Initializing architecture framework with random weights...")
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(config)
    # Resize embeddings to accommodate the newly added POS tags
    model.resize_token_embeddings(len(tokenizer))

    # 3. Setup and Sort Dataset
    print("Loading and sorting dataset by difficulty score...")
    full_dataset = load_dataset("json", data_files="data/babylm_english_with_pos.jsonl", split="train")
    
    # Sort the baseline dataset from easy to hard
    sorted_dataset = full_dataset.sort("score")

    block_size = 32

    # 4. Tokenization Subroutines
    def tokenize_pos(examples):
        # Remove spaces so special tokens match exactly without 'Ġ' space tokens
        cleaned_pos = [str(text).replace(" ", "") for text in examples["pos"]]
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [text + "<|endoftext|>" for text in cleaned_pos]
        return tokenizer(texts_with_eos)

    def tokenize_text(examples):
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["text"]]
        return tokenizer(texts_with_eos)

    # Group texts to maximize compute efficiency and eliminate padding waste
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder to ensure exact block sizes
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of max_length
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # EXPERIMENT B: FADING POS CURRICULUM (10 Epochs equivalent)
    # =========================================================================
    print("\n=== PREPARING FADING POS CURRICULUM DATASETS ===")
    
    # Define the decaying ratio of POS vs Text across 10 virtual epochs
    pos_ratios = [1.0, 0.89, 0.78, 0.67, 0.56, 0.44, 0.33, 0.22, 0.11, 0.0]
    epoch_datasets = []

    for epoch_idx, ratio in enumerate(pos_ratios, start=1):
        print(f"Processing Virtual Epoch {epoch_idx} Data (POS Ratio: {ratio*100:.0f}%)...")
        
        if epoch_idx == 1:
            # Epoch 1: 100% POS, strictly sorted by difficulty (No shuffling)
            ds = sorted_dataset.map(tokenize_pos, batched=True, remove_columns=full_dataset.column_names)
            ds = ds.map(group_texts, batched=True)
            epoch_datasets.append(ds)
        
        else:
            # Epoch 2-10: Mixed ratios, shuffled at sentence level
            shuffled_ds = sorted_dataset.shuffle(seed=42 + epoch_idx)
            split_idx = int(len(shuffled_ds) * ratio)
            
            # Slice the index to split POS and Text subsets
            pos_subset = None
            text_subset = None
            
            if split_idx > 0:
                pos_subset = shuffled_ds.select(range(split_idx)).map(
                    tokenize_pos, batched=True, remove_columns=full_dataset.column_names
                )
            
            if split_idx < len(shuffled_ds):
                text_subset = shuffled_ds.select(range(split_idx, len(shuffled_ds))).map(
                    tokenize_text, batched=True, remove_columns=full_dataset.column_names
                )
            
            # Recombine the slices
            if pos_subset and text_subset:
                combined_ds = concatenate_datasets([pos_subset, text_subset])
            elif pos_subset:
                combined_ds = pos_subset
            else:
                combined_ds = text_subset
                
            # Group into 256-token blocks
            grouped_ds = combined_ds.map(group_texts, batched=True)
            
            # Final block-level shuffle to evenly distribute POS blocks and Text blocks
            grouped_ds = grouped_ds.shuffle(seed=100 + epoch_idx)
            epoch_datasets.append(grouped_ds)

    print("\nConcatenating all virtual epochs into a single Master Curriculum Dataset...")
    # The master dataset naturally contains 10x the data volume of a single epoch
    master_dataset = concatenate_datasets(epoch_datasets)

    print("\n=== STARTING TRAINING ===")
    
    fading_args = TrainingArguments(
        output_dir="./babylm_fading_pos_english_output",
        num_train_epochs=1, # We train for 1 epoch over the 10x master dataset
        per_device_train_batch_size=128,
        dataloader_num_workers=4,
        save_strategy="steps",
        save_steps=5000,
		save_total_limit=1,
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    # Use CurriculumTrainer (SequentialSampler) so it reads the master dataset exactly in order
    trainer = CurriculumTrainer(
        model=model,
        args=fading_args,
        data_collator=data_collator,
        train_dataset=master_dataset,
    )
    
    trainer.train()
    trainer.save_model("./models/babylm_model_english_fading_pos_nospace")
    print("Fading POS Curriculum Experiment completed successfully.")

if __name__ == "__main__":
    main()