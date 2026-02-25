import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used in transformer encoder blocks
    Applies two linear layers with a GELU activation and dropout in between
    """

    def __init__(self,config):
        """
        Args:
            hidden_size (int): Input and output embedding size (d_model)
            middle_dim (int): Hidden layer size in the FFN (default: 4 * hidden_size in BERT)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size: int = config["hidden_size"]
        self.middle_dim: int = config["hidden_size"] * 4
        self.hidden_dropout_prob: float = config["dropout"]
        
        # Expand dimensionality (hidden_size → middle_dim)
        self.fc1 = torch.nn.Linear(self.hidden_size, self.middle_dim)

        # Project back to original size (middle_dim → hidden_size)
        self.fc2 = torch.nn.Linear(self.middle_dim, self.hidden_size)

        # Dropout applied after activation to prevent overfitting
        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)

        # GELU activation function (used in BERT)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        """
        Forward pass of the feed-forward network
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        
        # First linear transformation + GELU activation
        out = self.activation(self.fc1(x))

        # Apply dropout, then project back to original hidden size
        out = self.fc2(self.dropout(out))

        return out