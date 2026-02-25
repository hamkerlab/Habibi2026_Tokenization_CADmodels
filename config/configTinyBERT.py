import os
import argparse

class ConfigTinyBERT():

    def __init__(self, phase: str):
        self.phase = phase
        self.is_train = phase == "train"


        self.set_config_model()
        self.set_config_WordPiece()
        self.set_config_DeepCAD()
        
    def set_config_model(self):
        self.hidden_size = 128
        self.n_layers = 2
        self.heads = 2
        self.dropout = 0.1

    def set_config_WordPiece(self):
        """
        WordPiece model configuration
        """

        self.seq_len_naive = 1024
        self.seq_len_dc = 308

    def set_config_DeepCAD(self):
        """
        DeepCAD model configuration
        """

        self.n_args = 16
        self.args_dim = 256
        self.n_commands = 6  # line, arc, circle, EOS, SOS

        self.max_n_loops = 6
        self.max_n_curves = 15
        self.max_total_len = 60

        self.loss_cmd_weight = 1.0
        self.loss_args_weight = 2.0

        self.augment = True
        self.use_group = False
        self.max_num_groups = 30
        self.num_workers = 0

        return 


