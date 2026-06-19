from transformers import PreTrainedTokenizerFast

def main():
    # Load the newly trained tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_joint_tokenizer.json")
    
    # Add the special padding token
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    # 1. Test bilingual text (English first, Vietnamese second)
    test_sentence = "I can't see anything. \n\n Tôi không thể thấy gì. [SPLIT]"
    tokens = tokenizer.tokenize(test_sentence)
    print("Text tokens:", tokens)

    # 2. Test POS tags (English POS first, Vietnamese POS second)
    test_pos = "PRON AUX PART VERB PRON PUNCT \n\n PRON ADV VERB PRON PUNCT [SPLIT]"
    pos_tokens = tokenizer.tokenize(test_pos)
    print("POS tokens:", pos_tokens)

if __name__ == "__main__":
    # Execute the main program logic
    main()