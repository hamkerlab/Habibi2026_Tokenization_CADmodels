import math
import torch
import torch.nn as nn


class PositionalEncodingLUT(nn.Module):
    """
    Learned positional encoding using a lookup table

    Args:
        hidden_size (int): Dimensionality of embeddings
        dropout (float): Dropout probability
        max_len (int): Maximum sequence length
    """
    
    def __init__(self, hidden_size, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()

        # Initialize dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a tensor of position indices
        position = torch.arange(0, max_len, dtype=torch.long)
        # Register the position tensor as a buffer
        self.register_buffer('position', position)
        
        # Create an embedding layer to learn a vector for each position index
        self.pos_embed = nn.Embedding(max_len, hidden_size)
   
    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape [B, S, H]
        Returns:
            Tensor: Input embeddings plus positional embeddings [B, S, H]
        """
        
        B, S, H = x.size()

        # Slice positions to match input sequence length
        pos = self.position[:S]              # [S]
        pos_embed = self.pos_embed(pos)      # [S,H]
        pos_embed = pos_embed.unsqueeze(0)   # [1,S,H]
        
        # Add positional embeddings to input and apply dropout
        return x + pos_embed