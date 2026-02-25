import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR 
from torch.utils.tensorboard import SummaryWriter

class BERTTrainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader=None,
                 lr=None,
                 device="cuda",
                 results_dir=None,
                 total_epochs=None,
                 warmup_epochs=None,
                 patience=None,
                 min_delta=0.0
                 ):
        """
        Initialize the BERTTrainer

        Args:
            model: The BERT model to train
            train_dataloader: DataLoader for training
            val_dataloader: DataLoader for validation (optional)
            lr: Learning rate for Adam optimizer
            device: CUDA or CPU
            results_dir: Directory to save checkpoints & metrics
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs for learning rate scheduler
            patience: Early stopping patience
            min_delta: Minimum improvement in validation loss for early stopping
        """
        self.device = device  # Set the device (CPU or GPU)
        self.model = model.to(self.device)  # Move model to the specified device
        self.train_dataloader = train_dataloader  # Training DataLoader
        self.val_dataloader = val_dataloader  # Validation DataLoader (optional)

        # Optimizer & Loss
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)  # AdamW optimizer
        self.criterion = CrossEntropyLoss(ignore_index=-100)  # Loss function
        
        # Path for saving results
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)  # Create results directory if it doesn't exist
        
        # Metrics to track during training
        self.train_losses = []  # List to store training losses
        self.val_losses = []  # List to store validation losses
        self.lr_history = []  # List to store learning rate history

        # TensorBoard writer for logging
        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "runs"))

        # Create learning rate schedulers
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-5,
            last_epoch=-1
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_epochs]  
        )

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")  # Initialize best validation loss
        self.no_improve_count = 0  # Counter for epochs without improvement
        
    def train(self, epochs):
        """
        Main training loop for the model.

        Args:
            epochs: Number of epochs to train the model
        """
        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
            
            # Train for one epoch and record the training loss
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.writer.add_scalar("Training Loss", train_loss, epoch)

            self.scheduler.step()  # Update the learning rate

            # Learning rate tracking
            current_lr = self.scheduler.get_last_lr()[0]
            self.lr_history.append(current_lr)
            self.writer.add_scalar("LearningRate", current_lr, epoch)
            
            if self.val_dataloader:
                # Evaluate on the validation set and record the validation loss
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                self.writer.add_scalar("Validation Loss", val_loss, epoch)

                # Early stopping logic
                if self.patience is not None:
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.no_improve_count = 0
                        # Save the best model
                        best_model_path = os.path.join(self.results_dir, "best_model.pt")
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "train_losses": self.train_losses,
                            "val_losses": self.val_losses,
                            "lr_history": self.lr_history
                        }, best_model_path)
                        print(f"Best model saved at epoch {epoch + 1}")
                    else:
                        self.no_improve_count += 1
                        print(f"No improvement for {self.no_improve_count} epoch(s)")
                        if self.no_improve_count >= self.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break

            # Save model checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(self.results_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "lr_history": self.lr_history
                }, checkpoint_path)
         
        self.save_results()  # Save final results
        self.writer.close()  # Close TensorBoard writer
        print(f"Final model checkpoint saved.")

    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss for the epoch
        """
        self.model.train()  # Set model to training mode
        total_loss = 0  # Initialize total loss for the epoch
        loop = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(loop):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # Ensure token_type_ids exists
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"]).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)

            # Extract loss from model outputs
            if isinstance(outputs, dict):
                loss = outputs.get('loss') or outputs.get('loss_value')
                if loss is None:
                    raise ValueError("Neither 'loss' nor 'loss_value' found in model outputs!")
            else:
                raise TypeError("Model output is not a dictionary. Expected a dict with a 'loss' key")

            # Backpropagation
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update model parameters
            
            # Track loss
            total_loss += loss.item() 
            loop.set_postfix(loss=loss.item())  # Update progress bar with current loss

        avg_loss = total_loss / len(self.train_dataloader)  # Calculate average loss
        print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")

        return avg_loss  # Return average loss for the epoch

    def evaluate(self):
        """
        Evaluate the model on the validation set.

        Returns:
            avg_loss: Average validation loss
        """
        if self.val_dataloader is None:
            print("No validation data provided")
            return None

        self.model.eval()  # Set model to evaluation mode
        total_loss = 0  # Initialize total loss for validation

        with torch.no_grad():  # Disable gradient calculation
            loop = tqdm(self.val_dataloader, desc="Validation")

            for batch in loop:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                if "token_type_ids" not in inputs:
                    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"]).to(self.device)

                outputs = self.model(**inputs)

                if isinstance(outputs, dict):
                    loss = outputs.get("loss") or outputs.get("loss_value")
                else:
                    raise ValueError("Loss value is missing")

                total_loss += loss.item()  # Accumulate validation loss

        avg_loss = total_loss / len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0  # Calculate average loss

        print(f"Validation Loss: {avg_loss:.4f}")

        return avg_loss  # Return average validation loss

    def save_results(self):
        """
        Save training and validation losses to a JSON file.
        """
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses if self.val_losses else None,
            "lr_history": self.lr_history
        }

        results_path = os.path.join(self.results_dir, "metrics.json")

        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)  # Save metrics as JSON
        print(f"Results saved to {results_path}")  # Confirm results saved
