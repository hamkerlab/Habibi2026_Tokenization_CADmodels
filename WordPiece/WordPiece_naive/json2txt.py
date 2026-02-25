import os
import json
import re
import numpy as np
from joblib import Parallel, delayed

class json2txt:
    def __init__(self, base_dir, output_dir):
        """
        Args:
            base_dir (str): Directory containing json files
            output_dir (str): Directory where train, val, test and corpus.txt will be saved
        """

        self.base_dir = base_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.corpus_path = os.path.join(self.output_dir, "corpus.txt")

        self.num_regex = re.compile(r'(?<![A-Za-z0-9_+\-/])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_+\-/])')

    def _flatten_dict(self, nested_dict, parent_key=""):
        """
        Recursively flatten a nested dictionary
        """
        flattened_items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                flattened_items.extend(self._flatten_dict(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened_items.extend(self._flatten_dict(item, f"{new_key}[{i}]"))
                    else:
                        flattened_items.append(f"{new_key}[{i}]:{item}")
            else:
                flattened_items.append(f"{new_key}:{value}")

        return flattened_items
    
    def _process_json(self, file_path):
        """
        Process a single JSON file and return a list of tokens
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract sequence and entities
        sequence = data.get("sequence", [])
        entities = data.get("entities", {})

        flat_sequence = []

        for step in sequence:
            entity_id = step.get("entity", "")
            entity = entities.get(entity_id, {})

            flattened_step = self._flatten_dict(step, "sequence")
            flattened_entity = self._flatten_dict(entity, "entity")

            combined = " ".join(flattened_step + flattened_entity)
            flat_sequence.append(combined)

        # Flatten remaining fields
        remaining_data = {k: v for k, v in data.items() if k not in ["sequence", "entities"]}
        remaining_properties = self._flatten_dict(remaining_data)

        flattened_text = " ".join(flat_sequence + remaining_properties)

        # Tokenization: words and numbers
        tokens = re.findall(r'[A-Za-z0-9_+\-/]*[A-Za-z][A-Za-z0-9_+\-/]*|[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?',flattened_text)
        tokens = [t for t in tokens if t]

        return tokens
  
    def scale(self, files):
        """
        Compute normalization scale (min/max) from the training files
        """

        all_numbers = []

        for file_path in files:
            tokens = self._process_json(file_path)
            all_numbers.extend([float(x) for x in self.num_regex.findall(" ".join(tokens))])

        min_val = np.percentile(all_numbers, 1)
        max_val = np.percentile(all_numbers, 99)

        print(f"Normalization scale: min={round(min_val,4)}, max={round(max_val,4)}")

        return round(min_val,4), round(max_val,4)

    def _json2txt(self, file_path):
        """
        Process json and return list of tokens
        """
        return self._process_json(file_path)

    def _normalize(self, file_path, name, min_val, max_val):
        tokens = self._json2txt(file_path)
        content = " ".join(tokens)

        def normalize_match(match):
            val = float(match.group())   
            norm_val = 2 * (val - min_val) / (max_val - min_val) - 1

            # Clip to [-1,1]
            norm_val = max(-1, min(1, norm_val))
            return str(round(norm_val, 4))

        normalized_content = self.num_regex.sub(normalize_match, content)

        out_dir = os.path.join(self.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
        with open(out_path, "w") as f:
            f.write(normalized_content + "\n")
    
    def normalize(self, json_path, n_jobs=10):

        with open(json_path, "r") as f:
            split_data = json.load(f)

        json_filename = os.path.basename(json_path)

        dataset = {}
        for subset, ids in split_data.items():
            dataset[subset] = []
            for file_id in ids:
                json_file = os.path.join(self.base_dir, f"{file_id}.json")
                
                if os.path.basename(json_file) == json_filename:
                    continue

                dataset[subset].append(json_file)

        min_val, max_val = self.scale(dataset["train"])

        print(f"Normalization scale: min={min_val}, max={max_val}")

        for subset, files in dataset.items():
            Parallel(n_jobs=n_jobs, verbose=2)(
                delayed(self._normalize)(file_path, subset, min_val, max_val) for file_path in files
            )

        self.build_corpus() 
    
    def build_corpus(self):
        """
        Build a corpus file from all json files for training the WordPiece tokenizer
        """
        with open(self.corpus_path, "w") as corpus_file:
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(self.output_dir, split)

                for filename in sorted(os.listdir(split_dir)):
                    if filename.endswith(".txt"): # Process json files only

                        file_path = os.path.join(split_dir, filename)
                        with open(file_path, "r") as f:
                            corpus_file.write(f.read().strip() + "\n")


        print(f"Corpus file saved: {self.corpus_path}")




