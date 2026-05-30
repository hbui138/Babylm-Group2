import json

def sort_jsonl_by_score(input_filename, output_filename):
    print(f"Reading data from {input_filename}...")
    
    # Separate lists to hold data for Stage 1 and Stage 2
    stage1_data = []
    stage2_data = []
    
    # Read the JSONL file line by line
    with open(input_filename, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if line.strip():
                item = json.loads(line)
                if item.get("stage") == 1:
                    stage1_data.append(item)
                else:
                    stage2_data.append(item)
                
    print(f"Loaded {len(stage1_data) + len(stage2_data)} items. Sorting by complexity score per stage...")
    
    # Sort each stage independently based on the 'score' key in ascending order (easy to hard)
    stage1_data.sort(key=lambda x: x['score'])
    stage2_data.sort(key=lambda x: x['score'])
    
    print(f"Writing sorted data sequentially to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for item in stage1_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in stage2_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Sorting complete!")

if __name__ == "__main__":
    input_file = "bilingual_training_data.jsonl"
    output_file = "sorted_bilingual_training_data.jsonl"
    
    sort_jsonl_by_score(input_file, output_file)