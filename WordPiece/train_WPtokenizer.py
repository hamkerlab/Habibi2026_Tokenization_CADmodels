import argparse
from WPtokenizer import TokenizerTrainer
from config.cfg_data import PATHS

def main():
    parser = argparse.ArgumentParser("Train WordPiece tokenizer")
    
    parser.add_argument("--method", type=str, choices=["wp_naive", "wp_dc"], required=True, help="Which WordPiece method to train tokenizer for")
    parser.add_argument("--vocab_size", type=int, default=5000)

    args = parser.parse_args()

    if args.method == "wp_naive":
        corpus_path = PATHS.wp_naive_corpus
        save_dir = PATHS.wp_naive_tokenizer

    else:
        corpus_path = PATHS.wp_dc_corpus
        save_dir = PATHS.wp_dc_tokenizer


    trainer = TokenizerTrainer(
        corpus_dir = str(corpus_path),
        save_dir= str(save_dir),
        vocab_size=args.vocab_size
    )

    trainer.train()

if __name__ == "__main__":
    main()