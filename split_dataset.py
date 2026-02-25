import os
import json
import argparse
from random import seed, shuffle
from pathlib import Path
from config.cfg_data import PATHS

class DatasetSplit:
    """
    Class to split a dataset of json files into train, validation, and test sets
        
    Attributes:
        base_dir (str): Directory containing json files
        output_dir (str): Directory where train, val and test sets will be created
    """

    def __init__(self, base_dir: Path, split_file: Path):

        self.base_dir = Path(base_dir)
        self.split_file = Path(split_file)
        self.split_file.parent.mkdir(parents=True, exist_ok=True)
        
    def split_data(self, filename="train_val_test_split.json"):
        """
        Shuffle and split json files into train/val/test folders.
        """
        json_files = sorted(self.base_dir.glob("*.json"))
        data_ids = [p.stem for p in json_files]

        seed(42)
        shuffle(data_ids)

        train_size = int(0.8*len(data_ids))
        val_size = int(0.1*len(data_ids))
        test_size = len(data_ids) - train_size - val_size
        
        train = data_ids[:train_size]
        val = data_ids[train_size:train_size+val_size]
        test = data_ids[train_size + val_size:]

        data ={
            "train": train,
            "val": val,
            "test": test
        }

        self.split_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
     
        return self.split_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", required=True) 
    args = parser.parse_args()

    splitter = DatasetSplit(args.base_dir, args.output_dir)
    splitter.split_data()

if __name__ == "__main__":
    splitter = DatasetSplit(PATHS.cad_json, PATHS.splits)
    splitter.split_data()
