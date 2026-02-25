import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import random
import shutil

class BERTDataset(Dataset):
    """
    Dataset for loading .pt files and helper methods to build collator and dataloader
    """
    def __init__(self, data_dir, tokenizer_path=None, mlm_probability=0.15):
        """
        Args:
            data_dir (str): Directory of files
            tokenizer_path (str): Path to pretrained tokenizer
            mlm_probability (float): Probability for masking tokens
        """

        self.data_files = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith(".pt")
        ])

        self.tokenizer = None
        self.collator = None

        if tokenizer_path:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            self.collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
                return_tensors="pt"
            )

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):

        data = torch.load(self.data_files[idx])

        return {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"]
        }

    def dataloader(self, batch_size, shuffle=False):
        """
        Return DataLoader with MLM collator if tokenizer was given
        """

        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self.collator
        )







        