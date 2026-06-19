import json
from tokenizers import ByteLevelBPETokenizer

def joint_data_iterator(bilingual_filepath, english_filepath):
    # Stream data from the bilingual dataset
    print(f"Reading bilingual dataset: {bilingual_filepath}")
    with open(bilingual_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'vi_text' in data:
                yield data['vi_text']
            if 'pos_block' in data:
                yield data['pos_block']

    # Stream data from the pure English dataset
    print(f"Reading pure English dataset: {english_filepath}")
    with open(english_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data:
                yield data['text']

def main():
    # Initialize the ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Define UPOS tags strictly
    UPOS_TAGS = [
        "NOUN", "PROPN", "VERB", "ADJ", "PRON", 
        "DET", "ADV", "ADP", "CCONJ", "PUNCT", 
        "NUM", "PART", "INTJ", "X", "SCONJ", 
        "AUX", "SYM"
    ]

    # Construct the special tokens list
    special_tokens = ["<|endoftext|>", "[PAD]", "[SPLIT]"] + UPOS_TAGS

    print("Training Joint Tokenizer on combined datasets. Please wait...")

    # Train tokenizer using the combined iterator
    # Adjust vocab_size here if you want to compress it to 16000 instead of 32000
    tokenizer.train_from_iterator(
        joint_data_iterator("sorted_bilingual_training_data.jsonl", "english_only_training_data.jsonl"), 
        vocab_size=32000,
        min_frequency=2,
        special_tokens=special_tokens
    )
    
    # Save the output
    tokenizer.save("babylm_joint_tokenizer.json")
    print("Joint Tokenizer trained and saved successfully!")

if __name__ == "__main__":
    main()