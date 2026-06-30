import os
from transformers import AutoTokenizer

def main():
    # 1. Define the baseline model ID
    model_id = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"
    
    # 2. Define the path where your trained Exp 3 model is saved
    target_dir = "./models/babylm_exp3_aux_loss_dynamic_alpha_sorted"
    
    print(f"Loading official tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 3. Apply the exact same modification used during Exp 3 training
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        
    # 4. Save the tokenizer directly into the trained model's directory
    print(f"Saving tokenizer files to {target_dir}...")
    tokenizer.save_pretrained(target_dir)
    
    print("Done! You can now run your evaluation pipeline.")

if __name__ == "__main__":
    main()