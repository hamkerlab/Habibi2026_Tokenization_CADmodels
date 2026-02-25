import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import Embeddings
from layers.encoder import Encoder
from torch.utils.checkpoint import checkpoint


class BERT(nn.Module):
    """
    BERT backbone implementation
    """

    def __init__(self, config):
        
        """
        Args:
        vocab_size: Size of the vocabulary for token embeddings
        hidden_size: Dimensionality of the embeddings and hidden states
        n_layers: Number of Transformer encoder layers
        heads: Number of attention heads in multi-head attention
        dropout: Dropout rate applied in layers
        max_len: Maximum sequence length for positional embeddings
        n_segments: Number of segments (sentence A, sentence B)

        """

        super().__init__()
        
        # Model parameters
        self.hidden_size = config["hidden_size"] # hidden size
        self.n_layers = config["n_layers"] # Number of encoder layers
        self.heads = config["heads"] # Number of attention heads
        self.feed_forward_hidden = config["hidden_size"] * 4 # Dimensionality of feedforward hidden layers

        # Embedding layer combining token, positional, and segment embeddings
        self.embedding = Embeddings(config)

        # Stacking multiple transformer encoder layers
        self.encoder_blocks = torch.nn.ModuleList(
            [Encoder(
                config,
                config["hidden_size"],
                config["heads"],
                config["hidden_size"] * 4, 
                config["dropout"], 
                use_checkpoint=True 
            ) 
            for _ in range(self.n_layers)
            
        ])

    def forward(self, input_ids, segment_info, attention_mask=None):
        """
        Forward pass through BERT

        Args:
            x: Input token IDs (batch_size, seq_len)
            segment_info: Segment IDs (batch_size, seq_len), used to distinguish between two input
            attention_mask: (batch_size, seq_len) - Mask to ignore padding tokens

        Returns:
            x: (batch_size, seq_len, hidden_size) - Final hidden states
        """
        
        # Create attention mask

        # Mask to ensure attention ignores padding tokens
        # Shape: (batch_size, 1, 1, seq_len)
        mask = (attention_mask > 0).unsqueeze(1).unsqueeze(2) 

        # Embedding lookup
        x = self.embedding(input_ids, segment_info)
 
        # Pass through each encoder block
        for encoder in self.encoder_blocks:

            x = encoder(x, mask)
            
        return x

# Masked Language Model (MLM) Head
class MaskedLanguageModel(nn.Module):
    """
    The MLM head predicts masked tokens using a linear layer over the hidden states
    """

    def __init__(self, hidden_size, vocab_size): # weight tying
        """
        Args:
            hidden: Dimensionality of the input hidden states
            vocab_size: Size of the vocabulary for output predictions
        """
        super().__init__()

        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, hidden_states, labels=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - BERT outputs
            labels: (batch_size, seq_len) - Ground truth for MLM (-100 = ignore)

        Returns:
            logits: Predictions for each token in the vocabulary
            loss: Cross-entropy loss for MLM 
        """

        logits = self.linear(hidden_states) # Compute token logits
        output = {"logits": logits}

        if labels is not None:
            # Computing MLM loss, ignoring non-masked tokens (-100)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            output["loss"] = loss

        return output 

# BERT + MLM = BERTLM    
class BERTLM(nn.Module):
    """
    BERTLM combines the BERT backbone with the MLM head for the masked language modeling task
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration dict
        """
        super().__init__()

        # BERT backbone 
        self.bert = BERT(config)
        self.mask_lm = MaskedLanguageModel(
            config["hidden_size"], 
            config["vocab_size"]
        )

    def forward(self, input_ids, segment_ids=None,attention_mask=None,
                labels=None,**kwargs):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len)

        Returns:
            logits: Predictions for each token in the vocabulary
            loss: Cross-entropy loss for MLM 
        """
        
        # Get hidden states from BERT
        hidden_states = self.bert(input_ids,segment_ids, attention_mask=attention_mask)
        
        # Pass through MLM head
        output = self.mask_lm(hidden_states, labels)

        return output 