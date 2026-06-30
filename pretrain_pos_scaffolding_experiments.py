import os
import gc
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

num_proc_count = 1 # Adjust based on your CPU cores for optimal performance (e.g., 4 or 8)

# Custom Trainer to enforce strict sequential data feeding
class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        target_dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(target_dataset)

def main():
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    block_size = 32

    print("Loading Official 2026 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Inject UPOS tags
    upos_tags = ["NOUN", "PUNCT", "VERB", "PRON", "ADP", 
                "DET", "ADJ", "AUX", "ADV", "CCONJ", 
                "PROPN", "PART", "NUM", "SCONJ", "X", 
                "INTJ", "SYM"]
    tokenizer.add_special_tokens({'additional_special_tokens': upos_tags})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # Base configuration for model initialization later
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)

    print("Loading and sorting base dataset...")
    full_dataset = load_dataset("json", data_files="data/babylm_english_with_pos.jsonl", split="train")
    sorted_dataset = full_dataset.sort("score")
    column_names = full_dataset.column_names

    # Tokenization Subroutines
    def tokenize_pos(examples):
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["pos"]]
        return tokenizer(texts_with_eos)

    def tokenize_text(examples):
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["text"]]
        return tokenizer(texts_with_eos)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Reusable function to execute a single training run
    def run_experiment(experiment_name, master_dataset, output_dir):
        print(f"\n{'='*50}\nSTARTING EXPERIMENT: {experiment_name}\n{'='*50}")
        
        # Initialize a fresh model with random weights
        print("Initializing fresh model weights...")
        model = AutoModelForCausalLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=128,
            dataloader_num_workers=4,
            save_strategy="steps",
            save_steps=5000,
            save_total_limit=1,
            logging_steps=100,
            prediction_loss_only=True,
            fp16=True,
        )

        trainer = CurriculumTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=master_dataset,
        )
        
        trainer.train()
        trainer.save_model(f"./models/{experiment_name}")
        print(f"Finished {experiment_name}.")

        # Aggressive memory cleanup for Slurm node stability
        del model
        del trainer
        del training_args
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # STRATEGY 1: Linear Ratio with 10% Floor
    # =========================================================================
    print("\n[Strategy 1] Building Linear Floor Dataset...")
    linear_ratios = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    linear_epochs = []

    for epoch_idx, ratio in enumerate(linear_ratios, start=1):
        if epoch_idx == 1:
            ds = sorted_dataset.map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            ds = ds.map(group_texts, batched=True, num_proc=num_proc_count)
            linear_epochs.append(ds)
        else:
            shuffled_ds = sorted_dataset.shuffle(seed=42 + epoch_idx)
            split_idx = int(len(shuffled_ds) * ratio)
            
            pos_subset = shuffled_ds.select(range(split_idx)).map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            text_subset = shuffled_ds.select(range(split_idx, len(shuffled_ds))).map(tokenize_text, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            
            combined_ds = concatenate_datasets([pos_subset, text_subset])
            grouped_ds = combined_ds.map(group_texts, batched=True, num_proc=num_proc_count).shuffle(seed=100 + epoch_idx)
            linear_epochs.append(grouped_ds)

    ds_linear = concatenate_datasets(linear_epochs)
    run_experiment("babylm_strategy1_linear", ds_linear, "./output_strat1_linear")
    del linear_epochs, ds_linear; gc.collect()

    # =========================================================================
    # STRATEGY 2: Exponential/Logarithmic Decay with Floor
    # =========================================================================
    print("\n[Strategy 2] Building Exponential Decay Dataset...")
    # Fast drop off, trailing at 10% for the last half of training
    exp_ratios = [1.0, 0.60, 0.35, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10]
    exp_epochs = []

    for epoch_idx, ratio in enumerate(exp_ratios, start=1):
        if epoch_idx == 1:
            ds = sorted_dataset.map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            ds = ds.map(group_texts, batched=True, num_proc=num_proc_count)
            exp_epochs.append(ds)
        else:
            shuffled_ds = sorted_dataset.shuffle(seed=200 + epoch_idx)
            split_idx = int(len(shuffled_ds) * ratio)
            
            pos_subset = shuffled_ds.select(range(split_idx)).map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            text_subset = shuffled_ds.select(range(split_idx, len(shuffled_ds))).map(tokenize_text, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            
            combined_ds = concatenate_datasets([pos_subset, text_subset])
            grouped_ds = combined_ds.map(group_texts, batched=True, num_proc=num_proc_count).shuffle(seed=300 + epoch_idx)
            exp_epochs.append(grouped_ds)

    ds_exp = concatenate_datasets(exp_epochs)
    run_experiment("babylm_strategy2_exp", ds_exp, "./output_strat2_exp")
    del exp_epochs, ds_exp; gc.collect()

    # =========================================================================
    # STRATEGY 3: Chunking & 10% Fixed Memory Buffer
    # =========================================================================
    print("\n[Strategy 3] Building Chunked Dataset with 10% Memory Buffer...")
    
    # Split dataset into 10% permanent POS buffer and 90% active shifting data
    split_data = sorted_dataset.train_test_split(test_size=0.9, seed=42)
    reserve_ds = split_data['train']
    active_ds = split_data['test'].sort("score") # Ensure the 90% active data remains sorted by difficulty
    
    chunk_epochs = []
    
    for epoch_idx in range(1, 11): # 1 to 10
        # Calculate how much of the active dataset should remain POS
        # Epoch 1: 9/9 (100%), Epoch 10: 0/9 (0%)
        active_pos_ratio = (10 - epoch_idx) / 9.0 
        
        # Tokenize the permanent 10% reserve as POS
        tokenized_reserve = reserve_ds.map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
        
        if active_pos_ratio > 0:
            split_idx = int(len(active_ds) * active_pos_ratio)
            # The easier chunks remain POS
            active_pos = active_ds.select(range(split_idx)).map(tokenize_pos, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            # The harder chunks become Text
            if split_idx < len(active_ds):
                active_text = active_ds.select(range(split_idx, len(active_ds))).map(tokenize_text, batched=True, remove_columns=column_names, num_proc=num_proc_count)
                combined_ds = concatenate_datasets([tokenized_reserve, active_pos, active_text])
            else:
                # If split_idx == len(active_ds), 100% of the active data is POS
                combined_ds = concatenate_datasets([tokenized_reserve, active_pos])
        else:
            # Epoch 10: 0% active POS, 100% active Text
            active_text = active_ds.map(tokenize_text, batched=True, remove_columns=column_names, num_proc=num_proc_count)
            combined_ds = concatenate_datasets([tokenized_reserve, active_text])
            
        # Group and shuffle blocks to mix the Reserve POS and Active Text uniformly across the epoch
        grouped_ds = combined_ds.map(group_texts, batched=True, num_proc=num_proc_count).shuffle(seed=400 + epoch_idx)
        chunk_epochs.append(grouped_ds)

    ds_chunk = concatenate_datasets(chunk_epochs)
    run_experiment("babylm_strategy3_chunk", ds_chunk, "./output_strat3_chunk")
    del chunk_epochs, ds_chunk; gc.collect()

    print("\nAll Pre-training Strategies Completed Successfully.")

if __name__ == "__main__":
    main()