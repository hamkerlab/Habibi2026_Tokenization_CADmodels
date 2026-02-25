import argparse
from transformers import PreTrainedTokenizerFast
from config.cfg_data import PATHS
from tqdm import tqdm
import os
import torch
import numpy as np

class WPTokenizer:
    def __init__(self, tokenizer_dir, input_base_dir, output_base_dir, max_len=None):

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.max_len = max_len

        os.makedirs(self.output_base_dir, exist_ok=True)

    def tokenize(self, folder_name):
        
        input_dir = os.path.join(self.input_base_dir, folder_name)
        output_dir = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
        for filename in tqdm(txt_files, desc=f"{folder_name.upper()}"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Tokenize (no padding, truncation applied manually)
            tokenized = self.tokenizer(
                text.split(),
                return_attention_mask=True,
                return_special_tokens_mask=True,
                is_split_into_words=True,
                padding= False, 
                truncation=False
            )
        
            # Store sequence length
            seq_len = len(tokenized["input_ids"])
        
            # Extract word IDs (convert None → -1)
            word_ids = tokenized.word_ids()
            word_ids = [-1 if w is None else w for w in word_ids]

            # Truncate if seq_length > 1024
            if self.max_len and seq_len > self.max_len:
                tokenized["input_ids"] = tokenized["input_ids"][:self.max_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:self.max_len]
                word_ids = word_ids[:self.max_len]

            # Convert to tensors
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            word_ids = torch.tensor(word_ids, dtype=torch.long)

            # Save as .pt file
            output_file = os.path.join(output_dir, filename.replace(".txt", ".pt"))

            torch.save({
                "input_ids": input_ids, 
                "attention_mask": attention_mask,
                "word_ids": word_ids
            }, output_file)

    def tokenize_all(self):
        """
        Tokenize all: train, val, and test.
        """

        for x in ["train", "val", "test"]:
            self.tokenize(x)

def main():
    parser = argparse.ArgumentParser("Tokenize WordPiece datasets")
    parser.add_argument("--method", choices=["wp_naive", "wp_dc"], required=True)
    parser.add_argument("--max_len", type=int, default=None)
    args = parser.parse_args()

    if args.method == "wp_naive":
        tokenizer_dir = PATHS.wp_naive_tokenizer
        input_dir = PATHS.wp_naive
        output_dir = PATHS.wp_naive_tokenized
    else:
        tokenizer_dir = PATHS.wp_dc_tokenizer
        input_dir = PATHS.wp_dc
        output_dir = PATHS.wp_dc_tokenized

    tokenizer = WPTokenizer(
        tokenizer_dir=tokenizer_dir,
        input_base_dir=input_dir,
        output_base_dir=output_dir,
        max_len=args.max_len
    )

    tokenizer.tokenize_all()


if __name__ == "__main__":
    main()

