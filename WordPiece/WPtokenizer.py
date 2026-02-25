import os
from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.decoders import WordPiece as WordPieceDecoder
from transformers import PreTrainedTokenizerFast

class TokenizerTrainer:
    def __init__(self, corpus_dir, save_dir, vocab_size=None):
        """
        Args:
            corpus_dir (str): Directory containing text corpus files (.txt)
            save_dir (str): Directory to save the trained tokenizer
            vocab_size (int): Vocabulary size for the WordPiece tokenizer
        """

        self.corpus_dir = corpus_dir
        self.save_dir = save_dir
        self.vocab_size = vocab_size

        os.makedirs(save_dir, exist_ok=True)

    def batch_iterator(self):
        corpus_path = self.corpus_dir

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    def train(self):
        # Initialize the tokenizer
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

        # Use simple whitespace splitting for pre-tokenization
        tokenizer.pre_tokenizer = WhitespaceSplit()

        # Use WordPiece decoder (joins subwords with "##" prefix)
        tokenizer.decoder = WordPieceDecoder(prefix="##")

        # Trainer setup
        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        
        # Train tokenizer
        tokenizer.train_from_iterator(self.batch_iterator(), trainer)

        # Wrap with HuggingFace tokenizer
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

        # Save tokenizer
        wrapped_tokenizer.save_pretrained(self.save_dir)









