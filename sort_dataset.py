import json

def sort_jsonl_by_score(input_filename, output_filename):
    print(f"Reading data from {input_filename}...")
    
    # List to hold all parsed JSON objects
    dataset = []
    
    # Read the JSONL file line by line
    with open(input_filename, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip():
                dataset.append(json.loads(line))
                
    print(f"Loaded {len(dataset)} items. Sorting by complexity score...")
    
    # Sort the dataset based on the 'score' key in ascending order (easy to hard)
    dataset.sort(key=lambda x: x['score'])
    
    print(f"Writing sorted data to {output_filename}...")
    
    # Write the sorted objects sequentially to a new JSONL file
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for item in dataset:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Sorting complete!")

if __name__ == "__main__":
    input_file = "bilingual_training_data.jsonl"
    output_file = "sorted_bilingual_training_data.jsonl"
    
    sort_jsonl_by_score(input_file, output_file)