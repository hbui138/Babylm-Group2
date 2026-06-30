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

def run_experiment(model_id, full_dataset, pos_column, tags_list, exp_name):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Using POS column: {pos_column}")
    print(f"{'='*60}\n")

    # 1. Setup Official Tokenizer and Protect POS Tags
    print("Loading Official 2026 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Inject tags so BPE algorithm treats them as unbreakable single tokens
    tokenizer.add_special_tokens({'additional_special_tokens': tags_list})
    
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

    block_size = 32

    # 3. Tokenization Subroutines
    def tokenize_pos(examples):
        # Remove spaces so special tokens match exactly without 'Ġ' space tokens
        cleaned_pos = [str(text).replace(" ", "") for text in examples[pos_column]]
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [text + "<|endoftext|>" for text in cleaned_pos]
        return tokenizer(texts_with_eos)

    def tokenize_text(examples):
        # Append EOS token to mark sequence boundaries
        texts_with_eos = [str(text) + " <|endoftext|>" for text in examples["text"]]
        return tokenizer(texts_with_eos)

    # Group texts to maximize compute efficiency and eliminate padding waste
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

    print(f"Tokenizing datasets for {exp_name}...")
    
    # Process Phase 1 (POS) dataset
    tokenized_pos = full_dataset.map(tokenize_pos, batched=True, remove_columns=full_dataset.column_names)
    phase1_dataset = tokenized_pos.map(group_texts, batched=True)

    # Process Phase 2 (Text) dataset
    tokenized_text = full_dataset.map(tokenize_text, batched=True, remove_columns=full_dataset.column_names)
    text_dataset = tokenized_text.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize a single model to carry through both phases
    model_curriculum = initialize_fresh_model()

    # =========================================================================
    # PHASE 1: POS SCAFFOLDING (10 Epochs, POS only, Shuffled)
    # =========================================================================
    print(f"\n=== {exp_name} - RUNNING PHASE 1: POS SCAFFOLDING ===")
    
    shuffled_phase1_dataset = phase1_dataset.shuffle(seed=42)
    
    phase1_args = TrainingArguments(
        output_dir=f"./{exp_name}_phase1_output",
        num_train_epochs=10,
        per_device_train_batch_size=128,
        dataloader_num_workers=4,
        save_strategy="epoch",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer_phase1 = Trainer(
        model=model_curriculum,
        args=phase1_args,
        data_collator=data_collator,
        train_dataset=shuffled_phase1_dataset,
    )
    
    trainer_phase1.train()
    trainer_phase1.save_model(f"./models/{exp_name}_phase1")
    print(f"Phase 1 completed for {exp_name}.")

    # =========================================================================
    # PHASE 2: LEXICAL IMMERSION (10 Epochs, Text only, Shuffled)
    # =========================================================================
    print(f"\n=== {exp_name} - RUNNING PHASE 2: FULL LEXICAL IMMERSION ===")
    
    shuffled_phase2_dataset = text_dataset.shuffle(seed=42)

    phase2_args = TrainingArguments(
        output_dir=f"./{exp_name}_phase2_output",
        num_train_epochs=10,
        per_device_train_batch_size=128,
        dataloader_num_workers=4,
        save_strategy="epoch",
        logging_steps=100,
        prediction_loss_only=True,
        fp16=True,
    )

    trainer_phase2 = Trainer(
        model=model_curriculum, # Continuing with the syntax-primed model
        args=phase2_args,
        data_collator=data_collator,
        train_dataset=shuffled_phase2_dataset,
    )

    trainer_phase2.train()
    trainer_phase2.save_model(f"./models/{exp_name}_final")
    print(f"Phase 2 completed. {exp_name} finished successfully.\n")


def main():
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    
    print("Loading shared dataset...")
    full_dataset = load_dataset("json", data_files="data/babylm_english_with_pos.jsonl", split="train")

    # Define UPOS Tags (17 tags)
    upos_tags = [
        "NOUN", "PUNCT", "VERB", "PRON", "ADP", "DET", "ADJ", "AUX", 
        "ADV", "CCONJ", "PROPN", "PART", "NUM", "SCONJ", "X", "INTJ", "SYM"
    ]

    # Define Fine-grained English POS Tags (spaCy Penn Treebank tags)
    eng_pos_tags = [
        "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
        "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", 
        "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", 
        "VBZ", "WDT", "WP", "WP$", "WRB", "XX", "ADD", "AFX", "GW", "HYPH", 
        "NFP", ".", ",", "-LRB-", "-RRB-", "``", "''", ":", "$"
    ]

    # Run Experiment 1: UPOS
    run_experiment(
        model_id=model_id,
        full_dataset=full_dataset,
        pos_column="pos",
        tags_list=upos_tags,
        exp_name="babylm_10pos_10text_nospace"
    )

    # Run Experiment 2: English POS
    run_experiment(
        model_id=model_id,
        full_dataset=full_dataset,
        pos_column="english_pos",
        tags_list=eng_pos_tags,
        exp_name="babylm_10pos_10text_nospace_engpos"
    )

    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()