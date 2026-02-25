import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head scaled dot-product attention
    """

    def __init__(self, embed_size, num_heads, dropout=0.1):
        """
        Args:
            embed_size (int): Total embedding size (model dimension)
            num_heads (int): Number of attention weights
        """
        super().__init__()
        
        # Ensure that embed_size is divisible by num_heads
        assert embed_size % num_heads == 0, \
            "embed_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # Dimension per head

        # Dropout for regularizing attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections for queries, keys, and values
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Final output projection after concatenating all heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def attention(self, query, key, value, mask):
        """
        Compute scaled dot-product attention
        key: (batch, heads, seq_len, head_dim)

        Args:
            query: (batch, heads, seq_len, head_dim)
            key: (batch, heads, seq_len, head_dim)
            value: (batch, heads, seq_len, head_dim)
            mask

        Returns:
            context: (batch, heads, seq_len, head_dim)
        """

        # Compute scaled dot-product attention scores
        # Formula: QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(
                mask == 0, -1e9 
            )

        # Softmax over last dimension → attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply dropout for regularization
        weights = self.dropout(weights)

        # Multiply attention weights by values → context
        context = torch.matmul(weights, value)

        return context

    def forward(self, query, key, value, mask):
        """
        Forward pass for multi-head attention

        Args:
            query: (batch, seq_len, embed_size)
            key: (batch, seq_len, embed_size)
            value: (batch, seq_len, embed_size)
            mask

        Returns:
            context: (batch, seq_len, embed_size)
        """
       
        # Prepare attention mask shape
        if mask.dim() == 3:
            # Add extra dim for broadcasting: (batch, 1, 1, seq_len) 
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)

        # Linear projections for Q, K, V
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Reshape tensors for multi-head attention

        # Shape: (batch_size, num_heads, seq_len, head_dim)
        query = query.view(query.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Use gradient checkpointing to save memory
        context = checkpoint(self.attention, query, key, value, mask)

        # Reshape the context back to [batch_size, seq_len, embed_size]
        context = context.permute(0, 2, 1, 3).contiguous().view(
            context.shape[0], -1, self.num_heads * self.head_dim
        )
        
        # Final linear projection 
        context = self.fc_out(context)

        return context