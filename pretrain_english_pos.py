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
from datasets import load_dataset

# Custom Trainer to disable automatic shuffling for Curriculum Learning
class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        # Enforce strict sequential order to keep Phase 1 structure (Easy -> Hard POS)
        target_dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(target_dataset)
    
def main():
    # Setup Tokenizer (Ensure you point this to your English tokenizer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_joint_tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # Setup Model Architecture
    model_id = "BabyLM-community/babylm-baseline-10m-gpt2"
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    
    print("Initializing architecture framework with initialized random weights...")
    model = AutoModelForCausalLM.from_config(config)

    # Setup Dataset
    print("Loading and sorting dataset by difficulty score...")
    full_dataset = load_dataset("json", data_files="data/english_only_training_data_with_score.jsonl", split="train")
    
    # Sort the dataset from easy to hard based on the calculated score
    sorted_dataset = full_dataset.sort("score")

    block_size = 256

    # Tokenization subroutines mapping to the new JSON keys
    def tokenize_pos(examples):
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["pos"]]
        return tokenizer(texts_with_eos)

    def tokenize_text(examples):
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["text"]]
        return tokenizer(texts_with_eos)

    # Group texts to maximize compute efficiency
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================================================================
    # PHASE 1: SCAFFOLDING BOOTSTRAP (1 Epoch, POS only, Easy to Hard)
    # =========================================================================
    print("\n=== RUNNING PHASE 1: POS SCAFFOLDING (CURRICULUM LEARNING) ===")
    
    # Tokenize the dataset using the POS column
    tokenized_phase1 = sorted_dataset.map(tokenize_pos, batched=True, remove_columns=full_dataset.column_names)
    phase1_dataset = tokenized_phase1.map(group_texts, batched=True)

    phase1_args = TrainingArguments(
        output_dir="./babylm_phase1_english_with_pos_output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    # Use CurriculumTrainer to prevent shuffling and maintain the score-based sorting
    trainer_phase1 = CurriculumTrainer(
        model=model,
        args=phase1_args,
        data_collator=data_collator,
        train_dataset=phase1_dataset,
    )
    
    trainer_phase1.train()
    trainer_phase1.save_model("./babylm_phase1_english_with_pos")
    print("Phase 1 completed.")

    # =========================================================================
    # PHASE 2: LEXICAL IMMERSION (9 Epochs, Text only, Shuffled)
    # =========================================================================
    print("\n=== RUNNING PHASE 2: FULL LEXICAL IMMERSION (TEXT) ===")
    
    # Load model from the previous structural bootstrap checkpoint
    model = AutoModelForCausalLM.from_pretrained("./babylm_phase1_english_with_pos")
    
    # Tokenize using the actual natural language text stream
    tokenized_phase2 = sorted_dataset.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)
    phase2_dataset = tokenized_phase2.map(group_texts, batched=True)
    
    # Shuffle phase 2 sequence to maximize optimization robustness across epochs
    phase2_dataset = phase2_dataset.shuffle(seed=42)

    phase2_args = TrainingArguments(
        output_dir="./babylm_phase2_english_with_pos_output",
        num_train_epochs=9,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    # Use standard Trainer here because shuffling is desired for Phase 2
    trainer_phase2 = Trainer(
        model=model,
        args=phase2_args,
        data_collator=data_collator,
        train_dataset=phase2_dataset,
    )

    trainer_phase2.train()
    trainer_phase2.save_model("./babylm_pretrain_english_with_pos")
    print("Phase 2 completed.")

if __name__ == "__main__":
    main()