import os
import torch
from torch.utils.data import SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Custom Trainer to disable automatic shuffling for Curriculum Learning
class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        # Enforce strict sequential order to keep Phase 1 structure (Easy -> Hard POS)
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

    # Helper function to initialize a completely fresh model
    def initialize_fresh_model():
        print("Initializing architecture framework with random weights...")
        config = AutoConfig.from_pretrained(model_id)
        config.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(config)
        # Resize embeddings to accommodate the newly added POS tags
        model.resize_token_embeddings(len(tokenizer))
        return model

    # 2. Setup and Sort Dataset
    print("Loading and sorting dataset by difficulty score...")
    full_dataset = load_dataset("json", data_files="data/babylm_english_with_pos.jsonl", split="train")
    
    # Sort the dataset from easy to hard based on the calculated difficulty
    sorted_dataset = full_dataset.sort("score")

    block_size = 256

    # 3. Tokenization Subroutines
    def tokenize_pos(examples):
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["pos"]]
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

    print("Tokenizing datasets...")
    tokenized_pos = sorted_dataset.map(tokenize_pos, batched=True, remove_columns=full_dataset.column_names)
    phase1_dataset = tokenized_pos.map(group_texts, batched=True)

    tokenized_text = sorted_dataset.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)
    text_dataset = tokenized_text.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # EXPERIMENT A: PURE ENGLISH BASELINE (10 Epochs, Text only, Shuffled)
    # =========================================================================
    print("\n=== RUNNING EXPERIMENT A: ENGLISH BASELINE IMMERSION ===")
    model_baseline = initialize_fresh_model()

    # Standard shuffling for the baseline
    baseline_dataset = text_dataset.shuffle(seed=42)

    baseline_args = TrainingArguments(
        output_dir="./babylm_baseline_english_output",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer_baseline = Trainer(
        model=model_baseline,
        args=baseline_args,
        data_collator=data_collator,
        train_dataset=baseline_dataset,
    )

    trainer_baseline.train()
    trainer_baseline.save_model("./babylm_model_english_baseline")
    print("Experiment A (Baseline) completed.")

    # Free up memory before starting the next experiment
    del model_baseline
    del trainer_baseline
    torch.cuda.empty_cache()

    # =========================================================================
    # EXPERIMENT B - PHASE 1: SCAFFOLDING BOOTSTRAP (1 Epoch, POS only, Sorted)
    # =========================================================================
    print("\n=== RUNNING EXPERIMENT B - PHASE 1: POS SCAFFOLDING ===")
    model_curriculum = initialize_fresh_model()

    phase1_args = TrainingArguments(
        output_dir="./babylm_phase1_english_pos_output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    # Use CurriculumTrainer to prevent shuffling and enforce Easy->Hard order
    trainer_phase1 = CurriculumTrainer(
        model=model_curriculum,
        args=phase1_args,
        data_collator=data_collator,
        train_dataset=phase1_dataset,
    )
    
    trainer_phase1.train()
    trainer_phase1.save_model("./babylm_phase1_english_pos")
    print("Phase 1 completed.")

    # =========================================================================
    # EXPERIMENT B - PHASE 2: LEXICAL IMMERSION (9 Epochs, Text only, Shuffled)
    # =========================================================================
    print("\n=== RUNNING EXPERIMENT B - PHASE 2: FULL LEXICAL IMMERSION ===")
    
    # Shuffle phase 2 sequence to maximize optimization robustness
    phase2_dataset = text_dataset.shuffle(seed=42)

    phase2_args = TrainingArguments(
        output_dir="./babylm_phase2_english_pos_output",
        num_train_epochs=9,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    # Standard Trainer enables normal shuffling for text immersion
    trainer_phase2 = Trainer(
        model=model_curriculum,
        args=phase2_args,
        data_collator=data_collator,
        train_dataset=phase2_dataset,
    )

    trainer_phase2.train()
    trainer_phase2.save_model("./babylm_model_english_curriculum")
    print("Phase 2 completed. All experiments finished successfully.")

if __name__ == "__main__":
    main()