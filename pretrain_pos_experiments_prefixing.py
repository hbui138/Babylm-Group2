import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def main():
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    num_proc_count = 1  # Adjust based on your CPU cores for optimal performance

    # 1. Setup Official Tokenizer and Protect POS Tags
    print("Loading Official 2026 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Inject UPOS tags so BPE algorithm treats them as unbreakable single tokens
    upos_tags = ["NOUN", "PUNCT", "VERB", "PRON", "ADP", 
                 "DET", "ADJ", "AUX", "ADV", "CCONJ", 
                 "PROPN", "PART", "NUM", "SCONJ", "X", 
                 "INTJ", "SYM"]
    
    # Add POS tags alongside task prefix tokens
    tokenizer.add_special_tokens({
        'additional_special_tokens': upos_tags + ['<|pos_start|>', '<|text_start|>']
    })
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # 2. Initialize Fresh Model
    print("Initializing architecture framework with random weights...")
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(config)
    # Resize embeddings to accommodate the newly added POS tags
    model.resize_token_embeddings(len(tokenizer))

    # 3. Setup Dataset
    print("Loading dataset...")
    full_dataset = load_dataset("json", data_files="data/babylm_english_with_pos.jsonl", split="train")

    block_size = 32

    # 4. Tokenization Subroutines
    def format_prefix_task(examples):
        # Format: <|pos_start|> DET NOUN... <|text_start|> The cat... <|endoftext|>
        formatted_texts = []
        for pos_seq, text_seq in zip(examples["pos"], examples["text"]):
            combined = f"<|pos_start|> {pos_seq} <|text_start|> {text_seq} <|endoftext|>"
            formatted_texts.append(combined)
        return tokenizer(formatted_texts)

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

    # Note: For production, you should implement a custom data collator here 
    # to mask out the POS section with -100 so loss is only computed on the text.
    # For this baseline run, we use the standard LM collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # EXPERIMENT 2: TASK-PREFIXING PIPELINE
    # =========================================================================
    print("\n=== STARTING EXPERIMENT 2: TASK-PREFIXING ===")
    
    print("Processing Task-Prefixing Data...")
    tokenized_ds = full_dataset.map(
        format_prefix_task, 
        batched=True, 
        remove_columns=full_dataset.column_names, 
        num_proc=num_proc_count
    )
    
    # Apply the same grouping logic as the first file for consistency
    grouped_ds = tokenized_ds.map(
        group_texts, 
        batched=True, 
        num_proc=num_proc_count
    )

    print("\n=== STARTING TRAINING ===")
    
    training_args = TrainingArguments(
        output_dir="./output_exp2_task_prefix",
        num_train_epochs=10,
        per_device_train_batch_size=128,
        save_total_limit=1,
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,
        remove_unused_columns=False, # Important to keep all columns for the custom collator if implemented
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_ds,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model("./models/babylm_exp2_task_prefix")
    print("Experiment 2 Completed!")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    main()