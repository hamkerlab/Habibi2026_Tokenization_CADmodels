import torch
import torch.nn as nn 
from positional_encoding import *

class Embeddings(nn.Module):
    """
    Embeddings layer

    This layer combines token embeddings with positional embeddings and segment embeddings 
    to create the final embeddings

    """

    def __init__(self, config):
        """
        Initializes the Embeddings layer

        Args:
            config (object): Configuration object containing parameters
                - hidden_size (int): Size of the hidden embeddings
                - vocab_size (int): Size of the vocabulary
                - hidden_dropout_prob (float): Dropout probability for regularization
        """
        super().__init__()
        
        # Store config values as class attributes
        self.hidden_size: int = config["hidden_size"]
        self.vocab_size: int = config["vocab_size"]
        self.hidden_dropout_prob: float = config["dropout"]

        # Token embedding layer: converts token IDs into dense vectors of size "hidden_size"
        self.token_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        
        # Segment embedding layer: encodes sentence/segment information 
        self.segment_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=self.hidden_size
        )
        
        # Positional embedding layer: injects sequence order information
        self.positional_embeddings: PositionalEmbeddings = PositionalEmbeddings(config)
        
        # Dropout layer for regularization
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor = None, training: bool = False) -> torch.Tensor:
        """
        Forward pass of the Embeddings layer.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs of shape [batch_size, seq_len]
            segment_ids (torch.Tensor): Input tensor containing segment IDs
            training (bool): Whether the model is in training mode 

        Returns:
            torch.Tensor: Final embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # If no segment IDs are provided, default them to zeros (all tokens belong to the same segment)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        
        # Compute positional embeddings
        pos_info: torch.Tensor = self.positional_embeddings(input_ids)

        # Compute segment embeddings
        seg_info: torch.Tensor = self.segment_embeddings(segment_ids)
        
        # Compute token embeddings
        x: torch.Tensor = self.token_embeddings(input_ids)
        
        # Combine token + positional + segment embeddings
        x: torch.Tensor = x + pos_info + seg_info

        x: torch.Tensor = self.dropout(x)

        return x