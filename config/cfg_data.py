from pathlib import Path
from dataclasses import dataclass

DATA_ROOT = Path("")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class DatasetPaths:
    cad_json: Path = DATA_ROOT / "cad_json"
    splits: Path = DATA_ROOT / "train_val_test_split.json"

    wp_naive: Path = DATA_ROOT / "WordPiece-naive"
    wp_dc: Path = DATA_ROOT / "WordPiece-DC"
    deepcad: Path = DATA_ROOT / "DeepCAD"

    # corpus files
    wp_naive_corpus: Path = wp_naive / "corpus.txt"
    wp_dc_corpus: Path = wp_dc / "corpus.txt"
    
    # tokenizer
    wp_naive_tokenizer: Path = wp_naive / "tokenizer"
    wp_dc_tokenizer: Path = wp_dc / "tokenizer"

    # tokenized data 
    wp_naive_tokenized: Path = wp_naive / "tokenized_data"
    wp_dc_tokenized: Path = wp_dc / "tokenized_data"

PATHS = DatasetPaths()







