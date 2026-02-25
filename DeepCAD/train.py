"""
Main training script for the DeepCAD model using TinyBERT.

This script handles the configuration, data loading, model initialization, 
and training process for the DeepCAD project. It utilizes the BERT architecture 
for training on a specified dataset.

Usage:
    python train.py --data_root <data_root_path> --result_dir <result_directory> 
    --batch_size <batch_size> --nr_epochs <number_of_epochs> --lr <learning_rate> 
    --warmup_epochs <warmup_epochs> --patience <patience> --min_delta <min_delta> 
    --gpu_ids <gpu_ids> --seed <random_seed> 
"""

import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
from TinyBERT import BERT, BERTLM  
from torch.utils.data import DataLoader, Dataset
from trainer.trainerBERT import BERTTrainer
from datetime import datetime
from dataset import get_dataloader, mlm_collator
from cadlib.macro import *
from config.configTinyBERT import ConfigTinyBERT
from config.cfg_data import PATHS
from pathlib import Path

def main():
    """
    Main function to execute the training process.
    
    This function sets up the argument parser, initializes configurations, 
    prepares data loaders, initializes the model and trainer, and starts 
    the training process. It also logs the start and end time of the training.
    """

    base = ConfigTinyBERT("train")

    parser = argparse.ArgumentParser("DeepCAD Training")

    # paths
    parser.add_argument("--data_root", default=str(PATHS.deepcad))  # Path to the dataset root
    parser.add_argument("--result_dir", default=str(PATHS.deepcad / "results" / "training"))  # Directory to save results

    # training args
    parser.add_argument("--batch_size", type=int, default=100)  # Batch size for training
    parser.add_argument("--nr_epochs", type=int, default=500)  # Number of training epochs
    parser.add_argument("--lr", type=float, default=1e-4)  # Learning rate

    parser.add_argument("--warmup_epochs", type=int, default=10)  # Number of warmup epochs
    parser.add_argument("--patience", type=int, default=10)  # Early stopping patience
    parser.add_argument("--min_delta", type=float, default=0.001)  # Minimum change to qualify as an improvement

    parser.add_argument("--gpu_ids", type=str, default="0")  # GPU IDs to use for training
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")  # Seed for random number generation
    
    # DeepCAD params
    parser.add_argument("--max_total_len", type=int, default=base.max_total_len)  # Maximum total length of input sequences
    parser.add_argument("--max_n_loops", type=int, default=base.max_n_loops)  # Maximum number of loops
    parser.add_argument("--max_n_curves", type=int, default=base.max_n_curves)  # Maximum number of curves

    parser.add_argument("--n_commands", type=int, default=base.n_commands)  # Number of commands
    parser.add_argument("--n_args", type=int, default=base.n_args)  # Number of arguments
    parser.add_argument("--args_dim", type=int, default=base.args_dim)  # Dimension of arguments

    parser.add_argument("--hidden_size", type=int, default=base.hidden_size)  # Size of hidden layers
    parser.add_argument("--n_layers", type=int, default=base.n_layers)  # Number of layers in the model
    parser.add_argument("--heads", type=int, default=base.heads)  # Number of attention heads
    parser.add_argument("--dropout", type=float, default=base.dropout)  # Dropout rate

    parser.add_argument("--augment", action="store_true", default=base.augment, help="Enable data augmentation")  # Flag for data augmentation
    parser.add_argument("--use_group", action="store_true", default=base.use_group, help="Enable grouping (if supported)")  # Flag for grouping
    parser.add_argument("--max_num_groups", type=int, default=base.max_num_groups, help="Max number of groups")  # Maximum number of groups
    parser.add_argument("--num_workers", type=int, default=base.num_workers, help="DataLoader workers")  # Number of workers for DataLoader

    parser.add_argument("--loss_cmd_weight", type=float, default=base.loss_cmd_weight)  # Weight for command loss
    parser.add_argument("--loss_args_weight", type=float, default=base.loss_args_weight)  # Weight for argument loss

    args = parser.parse_args()  # Parse command line arguments

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)  # Set the GPU devices to use

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)  # Create result directory if it doesn't exist

    config = vars(args)  # Convert arguments to a dictionary

    config["loss_weights"] = {
        "loss_cmd_weight": float(getattr(args, "loss_cmd_weight", base.loss_cmd_weight)),  # Command loss weight
        "loss_args_weight": float(getattr(args, "loss_args_weight", base.loss_args_weight)),  # Argument loss weight
    }

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available

    # data loaders
    collator = mlm_collator(mask_prob=0.15)  # Initialize collator for data loading

    train_dataloader = get_dataloader("train", config, collate_fn=collator)  # Get training data loader
    val_dataloader = get_dataloader("val", config, collate_fn=collator)  # Get validation data loader

    # model
    bert = BERT(config, seq_len=config["max_total_len"])  # Initialize BERT model
    model = BERTLM(bert, config).to(device)  # Initialize BERTLM model and move to device

    # trainer
    trainer = BERTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=config["lr"],
        device=device,
        results_dir=config["result_dir"],
        total_epochs=config["nr_epochs"],
        warmup_epochs=config["warmup_epochs"],
        patience=config["patience"],
        min_delta=config["min_delta"],
    )  # Initialize trainer

    # train
    start_time = datetime.now()  # Record start time
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Print start time

    trainer.train(epochs=config["nr_epochs"])  # Start training

    end_time = datetime.now()  # Record end time
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Print end time
    print(f"Total training time: {end_time - start_time}")  # Print total training time


if __name__ == "__main__":
    main()  # Execute main function
import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import torch
from TinyBERT import BERT, BERTLM  
from torch.utils.data import DataLoader, Dataset
from trainer.trainerBERT import BERTTrainer
from datetime import datetime
from dataset import get_dataloader, mlm_collator
from cadlib.macro import *
from config.configTinyBERT import ConfigTinyBERT
from config.cfg_data import PATHS
from pathlib import Path

def main():

    base = ConfigTinyBERT("train")

    parser = argparse.ArgumentParser("DeepCAD Training")

    # paths
    parser.add_argument("--data_root", default=str(PATHS.deepcad))
    parser.add_argument("--result_dir", default=str(PATHS.deepcad / "results" / "training"))

    # training args
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nr_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    # DeepCAD params
    parser.add_argument("--max_total_len", type=int, default=base.max_total_len)
    parser.add_argument("--max_n_loops", type=int, default=base.max_n_loops)
    parser.add_argument("--max_n_curves", type=int, default=base.max_n_curves)

    parser.add_argument("--n_commands", type=int, default=base.n_commands)
    parser.add_argument("--n_args", type=int, default=base.n_args)
    parser.add_argument("--args_dim", type=int, default=base.args_dim)

    parser.add_argument("--hidden_size", type=int, default=base.hidden_size)
    parser.add_argument("--n_layers", type=int, default=base.n_layers)
    parser.add_argument("--heads", type=int, default=base.heads)
    parser.add_argument("--dropout", type=float, default=base.dropout)

    parser.add_argument("--augment", action="store_true", default=base.augment,help="Enable data augmentation")
    parser.add_argument("--use_group", action="store_true", default=base.use_group,help="Enable grouping (if supported)")
    parser.add_argument("--max_num_groups", type=int, default=base.max_num_groups,help="Max number of groups")
    parser.add_argument("--num_workers", type=int, default=base.num_workers,help="DataLoader workers")

    parser.add_argument("--loss_cmd_weight", type=float, default=base.loss_cmd_weight)
    parser.add_argument("--loss_args_weight", type=float, default=base.loss_args_weight)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args)

    config["loss_weights"] = {
    "loss_cmd_weight": float(getattr(args, "loss_cmd_weight", base.loss_cmd_weight)),
    "loss_args_weight": float(getattr(args, "loss_args_weight", base.loss_args_weight)),
}

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data loaders
    collator = mlm_collator(mask_prob=0.15)

    train_dataloader = get_dataloader("train", config, collate_fn=collator)
    val_dataloader = get_dataloader("val", config, collate_fn=collator)

    # model
    bert = BERT(config, seq_len=config["max_total_len"])
    model = BERTLM(bert, config).to(device)

    # trainer
    trainer = BERTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=config["lr"],
        device=device,
        results_dir=config["result_dir"],
        total_epochs=config["nr_epochs"],
        warmup_epochs=config["warmup_epochs"],
        patience=config["patience"],
        min_delta=config["min_delta"],
    )

    # train
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    trainer.train(epochs=config["nr_epochs"])

    end_time = datetime.now()
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")


if __name__ == "__main__":
    main()
