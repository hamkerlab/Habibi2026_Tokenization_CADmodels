import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .FFN import FeedForward
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    """
    Implements a transformer encoder block
    - Multi-head self-attention
    - Layer normalization + residual connections
    - Feed-forward network
    - Optional gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        config,
        hidden_size=128,
        heads=2,
        feed_forward_hidden=128 * 4,
        dropout=.1,
        use_checkpoint=False  
    ):
        """
        Args:
            hidden_size (int): Embedding size / model dimension
            heads (int): Number of attention heads
            feed_forward_hidden (int): Dimension of FFN hidden layer
                                Defaults to hidden_size * 4 (original BERT setting)
            dropout: Dropout rate
            use_checkpoint (bool): Enable gradient checkpointing for memory savings
            """
        super(Encoder, self).__init__()

        self.use_checkpoint = use_checkpoint
        
        # Layer normalization before attention & FFN
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Multi-head self-attention layer
        self.self_multihead = MultiHeadAttention(
            embed_size=hidden_size, 
            num_heads=heads, 
            dropout=dropout
        )

        # Position-wise Feed-Forward Network
        self.feed_forward = FeedForward(config)

        # Dropout layer applied after attention and FFN
        self.dropout = nn.Dropout(dropout)

    def _apply_layers(self, embeddings, mask):
        """
        Forward computation for one encoder block
        This function is separated so we can wrap it with torch.utils.checkpoint

        Args:
            embeddings: (batch_size, seq_len, hidden_size) 
            mask: Attention mask 
        
        Returns:
            encoded: (batch_size, seq_len, hidden_size)
        """
        
        # Apply self-attention
        attended = self.self_multihead(embeddings, embeddings, embeddings, mask)
        
        # Apply dropout to attention output
        attended = self.dropout(attended)
        
        # Residual connection + layer normalization
        interacted = self.norm1(attended + embeddings)

        # Feed-Forward Network + Residual + Norm

        # Pass through the position-wise FFN
        feed_forward_out = self.feed_forward(interacted)
        
        # Apply dropout to FFN output
        feed_forward_out = self.dropout(feed_forward_out)
        
        # Residual connection + layer normalization
        encoded = self.norm2(feed_forward_out + interacted)

        return encoded

    def check(self, embeddings, mask):
        """
        Wrapper for gradient checkpointing
        If enabled, it recomputes intermediate activations 
        during backpropagation to save memory at the cost of extra compute
        """
        if self.use_checkpoint:
            return checkpoint(self._apply_layers, embeddings, mask)
        else:
            return self._apply_layers(embeddings, mask)

    def forward(self, embeddings, mask):

        """
        Uses checkpointing if enabled
        """

        return self.check(embeddings, mask)