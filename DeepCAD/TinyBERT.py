import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import BERTEmbedding
from layers.encoder import Encoder
from cadlib.macro import CMD_ARGS_MASK
from cadlib.macro import *

class BERT(nn.Module):
    """
    BERT backbone implementation
    """

    def __init__(self, config, seq_len):

        """
        Args:
        hidden_size: Dimensionality of the embeddings and hidden states
        seq_len: 
        n_layers: Number of Transformer encoder layers
        heads: Number of attention heads in multi-head attention
        dropout: Dropout rate applied in layers
        use_group
        """

        super().__init__()

        self.hidden_size = config["hidden_size"]    
        self.n_layers = config["n_layers"]
        self.heads = config["heads"]
        self.dropout = config["dropout"]
        self.use_group = config["use_group"]

        # Embedding layer
        self.embedding = BERTEmbedding(config, seq_len, use_group=self.use_group)

        # Stack of Transformer blocks
        self.encoder_blocks = nn.ModuleList([
            Encoder(config,
                    config["hidden_size"], 
                    self.heads, self.hidden_size * 4, 
                    self.dropout, use_checkpoint=True)
                    for _ in range(self.n_layers)
            ])

    def forward(self, command, args, groups=None):

        # Input embedding
        x = self.embedding(command, args, groups) # [B, S, H]
        
        # Create token validity mask (1 for valid, 0 for padding where command == -1)
        valid_tokens = (command != -1).long()
        mask = valid_tokens.unsqueeze(1).unsqueeze(2) # Shape: (batch, 1, 1, seq_len)
        
        # Pass through stacked encoder blocks
        for encoder in self.encoder_blocks:
            x = encoder(x, mask)
            
        return x

class MaskedPredictionHead(nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        
        # Add 2 for special tokens
        self.n_commands = config["n_commands"] + 2 
        self.args_dim = config["args_dim"] + 2
        self.n_args = config["n_args"]

        # Prediction heads
        self.cmd_head = nn.Linear(hidden, self.n_commands)              # Predict command tokens
        self.args_head = nn.Linear(hidden, self.n_args * self.args_dim) # Predict args
        
        # Loss weighting
        self.weights = config["loss_weights"]
        
        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, out, tgt_commands, tgt_args, cmd_mask, arg_mask):
        B, S, _ = out.shape # [B, S, H]
        
        # Predict commands and args
        command_logits = self.cmd_head(out)  # Shape [S, N, n_commands]
        
        args_logits = self.args_head(out)  # Shape [B, S, n_args * args_dim]
        args_logits = args_logits.reshape(B, S, self.n_args, self.args_dim)
        
        # Command loss
        if cmd_mask.any():
            # Add +2 offset because target indices assume special token handling
            loss_cmd = F.cross_entropy(command_logits[cmd_mask],(tgt_commands[cmd_mask].long()+2),
            reduction="mean") 

        else:
            # No valid command tokens → return 0 loss
            loss_cmd = out.new_zeros(())
        
        # Args loss
        if arg_mask.any():
            masked_target_args = tgt_args[arg_mask].reshape(-1).long() + 2
            masked_pred_args = args_logits[arg_mask].reshape(-1, self.args_dim)
            
            loss_args = F.cross_entropy(masked_pred_args, masked_target_args,
            reduction="mean")

        else:
            loss_args = out.new_zeros(())

        # Apply weighting from config
        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        return {
            "loss_cmd": loss_cmd, 
            "loss_args": loss_args}
        

class BERTLM(nn.Module):
    def __init__(self, bert: BERT, config):

        super().__init__()
        self.bert = bert

        args_dim = config["args_dim"] + 1 
        self.prediction_head = MaskedPredictionHead(
            self.bert.hidden_size, config)

    def forward(self, command, args, tgt_commands=None,
                tgt_args=None,cmd_mask=None,
                arg_mask=None,groups=None, **kwargs):

        # Hidden states from BERT encoder
        hidden_states = self.bert(command, args, groups)
        
        # Training mode -> return losses
        if tgt_commands is not None and tgt_args is not None:
            loss_dict = self.prediction_head(hidden_states, 
                                            tgt_commands,
                                            tgt_args, cmd_mask, 
                                            arg_mask)

            return {
                "loss_cmd": loss_dict["loss_cmd"],
                "loss_args": loss_dict["loss_args"]
            }
            
        # Inference mode → return raw logits
        else:
            command_logits = self.prediction_head.cmd_head(hidden_states)
            args_logits = self.prediction_head.args_head(hidden_states)

            return {
                    "command_logits": command_logits,
                    "args_logits": args_logits 
                }


