import json

def load_jsonl(filepath):
    """Loads candidates from a JSONL file."""
    print(f"Loading data from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
