import os
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

class BERTTrainer:
    def __init__(self, 
                     model, 
                     train_dataloader, 
                     val_dataloader=None,
                     lr=1e-4,
                     device="cuda",
                     results_dir=None,
                     total_epochs=None,
                     warmup_epochs=None,
                     patience=None,
                     min_delta=0.0
                     ):
            """
            Initialize the BERTTrainer.

            Args:
            - model: The BERT model to train.
            - train_dataloader: DataLoader for training data.
            - val_dataloader: DataLoader for validation data (optional).
            - lr: Learning rate for the optimizer (default: 1e-4).
            - device: Device to run the model on, either "cuda" or "cpu" (default: "cuda").
            - results_dir: Directory path to save training results (default: None).
            - total_epochs: Total number of epochs for training (default: None).
            - warmup_epochs: Number of warmup epochs for learning rate scheduling (default: None).
            - patience: Number of epochs with no improvement after which training will be stopped (default: None).
            - min_delta: Minimum change in the monitored quantity to qualify as an improvement (default: 0.0).
            
            Initializes the optimizer, learning rate scheduler, and TensorBoard writer.
            Sets up lists to track training and validation losses, learning rate history, 
            and implements early stopping based on validation loss.
            """

            self.device = device
            self.model = model.to(self.device)
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            
            # optimizer
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            
            # output directory
            self.results_dir = results_dir
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Loss history
            self.train_cmd_losses = []
            self.train_args_losses = []
            self.train_total_losses = []

            self.val_cmd_losses = []
            self.val_args_losses = []
            self.val_total_losses = []

            self.lr_history = []
            
            # TensorBoard writer
            self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "runs"))
            
            # Scheduler
            self.warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor = 0.1,
                end_factor = 1.0,
                total_iters=warmup_epochs
            )

            self.cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max = total_epochs - warmup_epochs,
                eta_min = 1e-5,
                last_epoch = -1
            )

            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers = [self.warmup_scheduler, self.cosine_scheduler],
                milestones=[warmup_epochs]  
            )

            # Early stopping
            self.patience = patience
            self.min_delta = min_delta
            self.best_val_loss = float("inf")
            self.no_improve_count = 0


    def train(self, epochs):
        """Train the model for a given number of epochs."""
        
        for epoch in range(epochs):
            
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_cmd_loss, train_args_loss = self._train_epoch(epoch)
            train_total = train_cmd_loss + train_args_loss

            self.train_cmd_losses.append(train_cmd_loss)
            self.train_args_losses.append(train_args_loss)
            self.train_total_losses.append(train_total)

            self.writer.add_scalar("Loss/Train_Cmd", train_cmd_loss, epoch)
            self.writer.add_scalar("Loss/Train_Args", train_args_loss, epoch)
            self.writer.add_scalar("Loss/Train_Total", train_total, epoch)
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.lr_history.append(current_lr)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            if self.val_dataloader:
                val_cmd_loss, val_args_loss = self.evaluate()
                val_total = val_cmd_loss + val_args_loss

                self.val_cmd_losses.append(val_cmd_loss)
                self.val_args_losses.append(val_args_loss)
                self.val_total_losses.append(val_total)

                self.writer.add_scalar("Loss/Val_Cmd", val_cmd_loss, epoch)
                self.writer.add_scalar("Loss/Val_Args", val_args_loss, epoch)
                self.writer.add_scalar("Loss/Val_Total", val_total, epoch)

                # Early stopping
                if val_total < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_total
                    self.no_improve_count = 0
                    
                    best_path = os.path.join(self.results_dir, "best_model.pt")

                    # Save best model 
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    }, best_path)

                    print(f"Validation improved, saved best model")

                else:
                    self.no_improve_count += 1
                    print(f"No improvement for {self.no_improve_count} epoch(s)")

                    if self.no_improve_count >= self.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            
                            break

            if (epoch + 1) % 100 == 0:
                self.save_model(epoch + 1)

        self.save_model(epochs)
        print(f"Final model saved after {epochs} epochs !")

        self.save_results()

    def _train_epoch(self, epoch):
        """
        Train for one epoch
        """

        self.model.train()
        total_cmd_loss = 0.0
        total_args_loss = 0.0

        loop = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch in loop:
            # Move tensors to device
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) 
                      for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(command=inputs["command"], 
                                 args=inputs["args"],
                                 tgt_commands=inputs["tgt_commands"],
                                 tgt_args=inputs["tgt_args"],
                                 cmd_mask=inputs["cmd_mask"],
                                 arg_mask=inputs["arg_mask"])
            
            # Compute losses
            loss_cmd = outputs.get("loss_cmd", inputs["command"].new_zeros(()))
            loss_args = outputs.get("loss_args", inputs["command"].new_zeros(())) 
            
            loss_total = loss_cmd + loss_args
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Step optimizer
            self.optimizer.step()
            
            total_cmd_loss += loss_cmd.item() 
            total_args_loss += loss_args.item()  

            loop.set_postfix(cmd_loss=loss_cmd.item(),
                             arg_loss=loss_args.item(),
                             total_loss=loss_total.item())
   
        avg_cmd_loss = total_cmd_loss / len(self.train_dataloader)
        avg_args_loss = total_args_loss / len(self.train_dataloader)

        print(f"Train - Cmd Loss: {avg_cmd_loss:.4f} | Args Loss: {avg_args_loss:.4f}")
        return avg_cmd_loss, avg_args_loss

    def evaluate(self):
        """
        Evaluate the model on validation data.
        """
        if self.val_dataloader is None:
            print("No validation data provided.")
            return 0.0, 0.0

        self.model.eval()
        total_cmd_loss = 0
        total_args_loss = 0

        with torch.no_grad():

            loop = tqdm(self.val_dataloader, desc="Validation")

            for batch in loop:
                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) 
                        for k, v in batch.items()}

                outputs = self.model(command=inputs["command"],
                                     args=inputs["args"],
                                     tgt_commands=inputs["tgt_commands"],
                                     tgt_args=inputs["tgt_args"],
                                     cmd_mask=inputs["cmd_mask"],
                                     arg_mask=inputs["arg_mask"])

                loss_cmd = outputs.get("loss_cmd", inputs["command"].new_zeros(()))
                loss_args = outputs.get("loss_args", inputs["command"].new_zeros(()))

                total_cmd_loss += loss_cmd.item()
                total_args_loss += loss_args.item()

        avg_cmd_loss = total_cmd_loss / len(self.val_dataloader)
        avg_args_loss = total_args_loss / len(self.val_dataloader)
        
        print(f"Validation - Cmd Loss: {avg_cmd_loss:.4f} | Args Loss: {avg_args_loss:.4f}")
        
        return avg_cmd_loss, avg_args_loss

    def save_results(self):
        """Save training and validation losses."""

        metrics = {
            "train_cmd_losses": self.train_cmd_losses,
            "train_args_losses": self.train_args_losses,
            "train_total_losses": self.train_total_losses,
            "val_cmd_losses": self.val_cmd_losses,
            "val_args_losses": self.val_args_losses,
            "val_total_losses": self.val_total_losses,
            "lr_history": self.lr_history
        }

        results_path = os.path.join(self.results_dir, "metrics.json")

        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Results saved to {results_path}")

    def save_model(self, epoch):
        """Save the model checkpoint"""

        model_path = os.path.join(self.results_dir, f"bert_model_epoch_{epoch}.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()  
        }, model_path)
        
        print(f"Model saved at {model_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model from a checkpoint
        """

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if present
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load LR scheduler state if present
        if "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint {checkpoint_path} loaded successfully!")

        return checkpoint.get("epoch", None)
