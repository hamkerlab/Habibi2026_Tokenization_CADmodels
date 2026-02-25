import torch
import torch.nn as nn 
from positional_encoding import *
import sys
from cadlib.macro import *

class BERTEmbedding(nn.Module):
    """
    Command embeddings: Encodes the command tokens
    Argument embeddings: Encodes arguments associated with commands
    (Optional) Group embeddings: Encodes group information if enabled
    Positional encoding: Adds positional information for sequential data

    Args:
        config (dict)
        seq_len
        use_group (bool): Whether to include group embeddings
        group_len (int, optional): Number of group embeddings
    """

    def __init__(self, config, seq_len, use_group=False, group_len=None):
        super().__init__()
        
        # Command embeddings
        # +2 accounts for special tokens 
        self.command_embed = nn.Embedding(config["n_commands"] + 2, config["hidden_size"])
        
        # Argument embeddings
        # Each command has 16 arguments 
        n_args = N_ARGS
        args_dim = config["args_dim"] + 2 # +2 accounts for special tokens
        
        # Each argument token is mapped to a 64-dimensional embedding
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=1)

        # After embedding all arguments, we flatten them and project into the model dimension (hidden_size)
        self.embed_fcn = nn.Linear(64 * n_args, config["hidden_size"])
        
        # Group embeddings (optional)
        self.use_group = use_group
        if self.use_group:
            if group_len is None:
                group_len = config["max_num_groups"]
            self.group_embed = nn.Embedding(group_len + 2, config["hidden_size"])
        
        # Positional encoding
        self.pos_encoding = PositionalEncodingLUT(config["hidden_size"], max_len=seq_len+2)

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, commands, args, groups=None):
        """
        Forward pass for embedding the input sequences

        Args:
            commands: [B, S]
            args:     [B, S, n_args]
            groups (optional)

        Returns:
            src: Combined embeddings [B, S, H] 
        """

        B, S = commands.shape

        src = self.command_embed((commands + 2).long()) + \
            self.embed_fcn(self.arg_embed((args + 2).long()).view(B,S, -1)) # shift due to -1 PAD_VAL
            
        if self.use_group and groups is not None:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        src = self.dropout(src)

        return src