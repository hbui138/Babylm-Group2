import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer

def data_iterator(filepath):
    # Generator to stream text directly from the JSONL file to save RAM
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Yield the bilingual text block to train the BPE
            yield data['text_block']
            yield data['pos_block']

def main():
    # Initialize a standard BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Use ByteLevelBPETokenizer for better handling of raw text and special tokens
    tokenizer = ByteLevelBPETokenizer()

    UPOS_TAGS = [
        "NOUN", "PROPN", "VERB", "ADJ", "PRON", 
        "DET", "ADV", "ADP", "CCONJ", "PUNCT", 
        "NUM", "PART", "INTJ", "X", "SCONJ", 
        "AUX", "SYM"
    ]

    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SPLIT]"] + UPOS_TAGS

    # Configure the trainer
    trainer = BpeTrainer(
        vocab_size=16000, 
        special_tokens=special_tokens,
        show_progress=True,
    )

    print("Training BPE Tokenizer from scratch. This might take a minute...")

    # Train the tokenizer
    tokenizer.train_from_iterator(data_iterator("sorted_bilingual_training_data.jsonl"), trainer=trainer)

    # Save the trained tokenizer
    tokenizer.save("babylm_bilingual_tokenizer.json")
    print("Tokenizer trained and saved successfully!")

if __name__ == "__main__":
    main()