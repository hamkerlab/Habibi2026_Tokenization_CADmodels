import sys
import os
import argparse
import numpy as np
import random
from pathlib import Path
from datetime import datetime

project_root = ""
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import PreTrainedTokenizerFast

from dataset import BERTDataset
from TinyBERT import BERT, BERTLM
from trainer.trainerBERT import BERTTrainer
from config.configTinyBERT import ConfigTinyBERT
from config.cfg_data import PATHS


def set_seed(seed: int):
    """Make runs reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Model defaults (hidden_size, n_layers, heads, dropout, seq_len)
    cfg = ConfigTinyBERT(phase="train")

    parser = argparse.ArgumentParser("WordPiece TinyBERT Training")

    # Select which dataset/tokenizer to use
    parser.add_argument("--method", choices=["wp_naive", "wp_dc"], required=True)

    # Training hyperparameters (override as needed)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nr_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)

    # GPU/seed
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    # Optional path overrides (usually you won't need these)
    parser.add_argument("--data_root", type=str, default=None, help="tokenized_data root")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="tokenizer directory")
    parser.add_argument("--result_dir", type=str, default=None, help="output directory")

    args = parser.parse_args()

    # GPU + runtime settings
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "wp_dc":
        seq_len = cfg.seq_len_dc
        data_root = PATHS.wp_dc_tokenized
        tokenizer_path = PATHS.wp_dc_tokenizer
        result_dir = PATHS.wp_dc / "results" / "training"

    elif args.method == "wp_naive":
        seq_len = cfg.seq_len_naive
        data_root = PATHS.wp_naive_tokenized
        tokenizer_path = PATHS.wp_naive_tokenizer
        result_dir = PATHS.wp_naive / "results" / "training"

    else:
        raise ValueError(f"Unknown method: {args.method}")

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(data_root)

    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
    else:
        tokenizer_path = Path(tokenizer_path)

    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = Path(result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (for vocab size)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    vocab_size = tokenizer.vocab_size

    # Dataset / Dataloaders
    train_data = BERTDataset(
        data_dir=str(data_root / "train"),
        tokenizer_path=str(tokenizer_path),
    )
    val_data = BERTDataset(
        data_dir=str(data_root / "val"),
        tokenizer_path=str(tokenizer_path),
    )

    train_dataloader = train_data.dataloader(args.batch_size, shuffle=True)
    val_dataloader = val_data.dataloader(args.batch_size, shuffle=False)

    # Model setup
    model_config = {
        "hidden_size": cfg.hidden_size,
        "n_layers": cfg.n_layers,
        "heads": cfg.heads,
        "dropout": cfg.dropout,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }

    bert = BERT(model_config)
    model = BERTLM(model_config).to(device)

    # Trainer
    trainer = BERTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=args.lr,
        device=device,
        results_dir=str(result_dir),
        total_epochs=args.nr_epochs,
        warmup_epochs=int(args.warmup_epochs),
        patience=args.patience,
        min_delta=args.min_delta,
    )

    # Train
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Using:")
    print(" method      :", args.method)
    print(" data_root   :", data_root)
    print(" tokenizer   :", tokenizer_path)
    print(" result_dir  :", result_dir)
    print(" batch_size  :", args.batch_size)
    print(" lr          :", args.lr)
    print(" nr_epochs   :", args.nr_epochs)
    print(" device      :", device)

    trainer.train(epochs=args.nr_epochs)

    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")


if __name__ == "__main__":
    main()




    















