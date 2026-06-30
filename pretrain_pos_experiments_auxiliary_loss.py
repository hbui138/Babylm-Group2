import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    GPT2PreTrainedModel,
    GPT2Model
)
from transformers import TrainerCallback
from torch.utils.data import SequentialSampler


# Custom Callback to update alpha dynamically at the start of each epoch
class DynamicAlphaCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        # Current epoch is available in state.epoch (float, e.g., 0.0, 1.0)
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Define a decaying alpha schedule: High priority for POS at start, fading later
        alpha_schedule = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05] # Fixed alpha is 0.2 before
        
        if current_epoch < len(alpha_schedule):
            chosen_alpha = alpha_schedule[current_epoch]
        else:
            chosen_alpha = 0.05
            
        # Dynamically inject the new alpha directly into the model
        if hasattr(model, "alpha"):
            model.alpha = chosen_alpha
            print(f"\n[Callback] Epoch {current_epoch + 1}: Setting Auxiliary Loss Alpha to {chosen_alpha}")

class CurriculumTrainer(Trainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        # Enforce strict sequential order to read from Easy to Hard
        target_dataset = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(target_dataset)

# Custom GPT-2 model with dual heads: 
# 1. Language Modeling Head (predicts next word)
# 2. POS Tagging Head (predicts POS tag of the next word)
class BabyLMWithPOS(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        
        # Standard LM head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Auxiliary POS head
        self.pos_head = nn.Linear(config.n_embd, config.pos_vocab_size, bias=False)
        
        # Target layer for POS extraction (e.g., layer 4 of 12)
        self.pos_layer_idx = 4

        self.alpha = 1.0

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pos_labels=None,
        **kwargs
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        final_hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = lm_loss

            if pos_labels is not None:
                all_hidden_states = transformer_outputs.hidden_states
                middle_hidden_states = all_hidden_states[self.pos_layer_idx]
                
                pos_logits = self.pos_head(middle_hidden_states)

                shift_pos_logits = pos_logits[..., :-1, :].contiguous()
                shift_pos_labels = pos_labels[..., 1:].contiguous()
                
                pos_loss_fct = CrossEntropyLoss(ignore_index=-100)
                pos_loss = pos_loss_fct(shift_pos_logits.view(-1, self.config.pos_vocab_size), shift_pos_labels.view(-1))
                
                loss = lm_loss + (self.alpha * pos_loss)

        return {"loss": loss, "logits": lm_logits}


def main():
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    data_path = "data/babylm_english_with_pos.jsonl"
    block_size = 32
    num_proc_count = 4 # Increased for faster mapping
    
    # 1. Setup Official Tokenizer and POS mapping
    print("Loading Official 2026 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    upos_tags = ["NOUN", "PUNCT", "VERB", "PRON", "ADP", "DET", "ADJ", "AUX", 
                 "ADV", "CCONJ", "PROPN", "PART", "NUM", "SCONJ", "X", "INTJ", "SYM"]
    pos_to_id = {tag: i for i, tag in enumerate(upos_tags)}

    # 2. Initialize Fresh Model
    print("Initializing architecture framework with dual heads...")
    config = AutoConfig.from_pretrained(model_id)
    config.vocab_size = len(tokenizer)
    config.pos_vocab_size = len(upos_tags)
    
    model = BabyLMWithPOS(config)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Setup Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Sort the dataset based on the 'difficulty' field from Easy to Hard (comment if not needed)
    print("Sorting dataset by syntactic difficulty...")
    dataset = dataset.sort("score")

    # 4. Tokenization Subroutines
    def tokenize_with_pos_labels(examples):
        words_batch = [str(text).split() for text in examples["text"]]
        pos_batch = [str(pos).split() for pos in examples["pos"]]
        
        # Tokenize WITHOUT truncation or padding to allow for grouping later
        tokenized_inputs = tokenizer(
            words_batch, 
            is_split_into_words=True,
        )
        
        labels_batch = []
        for batch_idx, pos_tags in enumerate(pos_batch):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    if word_idx < len(pos_tags):
                        tag = pos_tags[word_idx]
                        label_ids.append(pos_to_id.get(tag, -100))
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                    
                previous_word_idx = word_idx
            
            # Append EOS token to separate documents when they get concatenated
            tokenized_inputs["input_ids"][batch_idx].append(tokenizer.pad_token_id)
            tokenized_inputs["attention_mask"][batch_idx].append(1)
            label_ids.append(-100)
            
            labels_batch.append(label_ids)
            
        tokenized_inputs["pos_labels"] = labels_batch
        tokenized_inputs["labels"] = [row.copy() for row in tokenized_inputs["input_ids"]]
        
        return tokenized_inputs

    def group_texts(examples):
        # Concatenate all lists across the batch
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder to ensure identical sequence lengths
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split into uniform chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def custom_collate(features):
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.tensor([f[key] for f in features])
        return batch

    print("Applying tokenization and POS alignment...")
    tokenized_ds = dataset.map(
        tokenize_with_pos_labels, 
        batched=True, 
        remove_columns=dataset.column_names, 
        num_proc=num_proc_count,
        desc="Tokenizing and Aligning"
    )

    print("Grouping texts into uniform blocks...")
    grouped_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        num_proc=num_proc_count,
        desc=f"Grouping texts into chunks of {block_size}"
    )

    # =========================================================================
    # EXPERIMENT 3: AUXILIARY LOSS PIPELINE
    # =========================================================================
    print("\n=== STARTING EXPERIMENT 3: AUXILIARY LOSS ===")
    os.makedirs("./models", exist_ok=True)

    training_args = TrainingArguments(
        output_dir="./output_exp3_aux_loss_dynamic_alpha_sorted",
        num_train_epochs=10,
        per_device_train_batch_size=128,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        fp16=True,
        remove_unused_columns=False, # Crucial: prevent Trainer from dropping 'pos_labels'
        dataloader_num_workers=4,
    )

    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=grouped_ds, # Use the grouped dataset
        data_collator=custom_collate,
        callbacks=[DynamicAlphaCallback()]  # Attach the dynamic alpha callback
    )

    trainer.train()
    trainer.save_model("./models/babylm_exp3_aux_loss_dynamic_alpha_sorted")
    print("Experiment 3 Completed!")

if __name__ == "__main__":
    main()