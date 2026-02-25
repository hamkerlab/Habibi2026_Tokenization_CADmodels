"""
This module defines the CADDataset class for loading and processing CAD data 
for training and evaluation of models. It utilizes PyTorch's Dataset and DataLoader 
to facilitate batch processing and augmentation of CAD vectors.

The code is based on the DeepCAD repository. For more information, visit:
https://github.com/rundiwu/DeepCAD
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import sys
from cadlib.macro import *

class CADDataset(Dataset):
    def __init__(self, phase, config):
        """
        Initialize the CADDataset.

        Args:
            phase (str): The phase of the dataset, e.g., "train", "val", or "test".
            config (dict): Configuration dictionary containing dataset parameters.
        """
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config["data_root"], phase)  # h5 data root
        self.phase = phase
        self.aug = config["augment"]  # Whether to apply augmentation
        
        # List all h5 files in the directory
        self.all_data = [os.path.splitext(f)[0] for f in os.listdir(self.raw_data) if f.endswith('.h5')]
    
        self.max_n_loops = config["max_n_loops"]         
        self.max_n_curves = config["max_n_curves"]           
        self.max_total_len = config["max_total_len"]
        self.size = 256  # Size of the input data

    def get_data_by_id(self, data_id):
        """Return dataset item by its ID.

        Args:
            data_id (str): The ID of the data item to retrieve.

        Returns:
            dict: The data item corresponding to the given ID.
        """
        idx = self.all_data.index(data_id)  # Find the index of the data_id
        return self.__getitem__(idx)  # Retrieve the item using its index

    def __getitem__(self, index):
        """Retrieve a data item by its index.

        Args:
            index (int): The index of the data item.

        Returns:
            dict: A dictionary containing the command, arguments, and ID of the data item.
        """
        data_id = self.all_data[index]  # Get the data ID from the list
        h5_path = os.path.join(self.raw_data, data_id + ".h5")  # Construct the file path
        
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]  # Load the CAD vector from the h5 file

        # Apply augmentation if in training phase
        if self.aug and self.phase == "train":
            command1 = cad_vec[:, 0]  # Extract commands
            ext_indices1 = np.where(command1 == EXT_IDX)[0]  # Find indices of external commands

            # Randomly decide to apply augmentation
            if len(ext_indices1) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec1 = np.split(cad_vec, ext_indices1 + 1, axis=0)[:-1]  # Split the CAD vector
                
                # Randomly select another data item for augmentation
                data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]  # Load the second CAD vector

                command2 = cad_vec2[:, 0]  # Extract commands from the second vector
                ext_indices2 = np.where(command2 == EXT_IDX)[0]  # Find external command indices
                ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]  # Split the second CAD vector
                
                # Randomly replace parts of the first external vector with the second
                n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
                new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                for i in range(len(old_idx)):
                    ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]  # Replace external commands

                # Concatenate the modified external vector
                sum_len = 0
                new_vec = []
                for i in range(len(ext_vec1)):
                    sum_len += len(ext_vec1[i])
                    if sum_len > self.max_total_len:
                        break
                    new_vec.append(ext_vec1[i])
                cad_vec = np.concatenate(new_vec, axis=0)  # Combine the new vector

        # Padding sequence to max_total
        pad_len = self.max_total_len - cad_vec.shape[0]  # Calculate padding length

        if pad_len > 0:
            pad_vec = np.full((pad_len, cad_vec.shape[1]), -1, dtype=cad_vec.dtype)  # Create padding vector
            cad_vec = np.concatenate([cad_vec, pad_vec], axis=0)  # Pad the CAD vector
        
        # Split commands and arguments
        command = cad_vec[:, 0]  # Extract commands
        args = cad_vec[:, 1:]  # Extract arguments

        # Convert to PyTorch tensors
        command = torch.tensor(command, dtype=torch.long)  # Convert commands to tensor
        args = torch.tensor(args, dtype=torch.long)  # Convert arguments to tensor

        return {"command": command, "args": args, "id": data_id}  # Return the data item
    
    def __len__(self):
        """Return the total number of sequences in the dataset.

        Returns:
            int: The total number of data items in the dataset.
        """
        return len(self.all_data)  # Return the length of the dataset


def get_dataloader(phase, config, shuffle=None, collate_fn=None):
    """
    Create a PyTorch dataloader for a given dataset phase.

    Args:
        phase (str): "train", "val", or "test".
        config (dict): Configuration dictionary with keys like batch_size, num_workers, data_root.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True for training.
        collate_fn: Custom collate function, e.g., mlm_collator.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified dataset phase.
    """
    is_shuffle = phase == 'train' if shuffle is None else shuffle  # Determine if shuffling is needed
    
    # Initialize dataset
    dataset = CADDataset(phase, config)  # Create an instance of the dataset

    # Create dataloader
    dataloader = DataLoader(dataset, 
                batch_size=config["batch_size"], 
                shuffle=is_shuffle, 
                num_workers=config["num_workers"],
                worker_init_fn=np.random.seed(), 
                collate_fn=collate_fn)  # Create DataLoader

    return dataloader  # Return the DataLoader


def mlm_collator(mask_prob=.15, mask_token=-2, pad_val=-1, mask_arg_val=-2):
    """
    Creates a collator function for MLM training:
    - Randomly masks commands and arguments with probability "mask_prob"
    - Returns both masked inputs and original targets.

    Args:
        mask_prob (float): Probability of masking a token.
        mask_token (int): Token used for masking commands.
        pad_val (int): Padding value for commands.
        mask_arg_val (int): Token used for masking arguments.

    Returns:
        function: A collate function for masking commands and arguments.
    """
    def collate_fn(batch):
        # Stack commands/args into a batch
        commands = torch.stack([item["command"] for item in batch])  # Stack commands
        args = torch.stack([item["args"] for item in batch])  # Stack arguments
        
        # Targets 
        tgt_commands = commands.clone()  # Clone commands for targets
        tgt_args = args.clone()  # Clone arguments for targets
        
        # Valid command positions (ignore padding tokens)
        valid_cmd_mask = (commands != pad_val)  # Create mask for valid commands
        
        # Random mask for commands
        random_mask = torch.rand(commands.shape, device=commands.device) < mask_prob  # Generate random mask
        rand_cmd_mask = random_mask & valid_cmd_mask  # Combine with valid command mask 
        
        # Apply command mask
        masked_commands = commands.clone()  # Clone commands for masking
        masked_commands[rand_cmd_mask] = mask_token  # Apply mask to commands
        
        # Valid arg positions (ignore padding tokens)
        valid_args_mask = (args != pad_val)  # Create mask for valid arguments
        
        # Random mask for args
        rand_arg_mask = (torch.rand(args.shape, device=args.device) < mask_prob) & valid_args_mask  # Generate random mask for args
        
        # Apply arg mask
        masked_args = args.clone()  # Clone arguments for masking
        masked_args[rand_arg_mask] = mask_arg_val  # Apply mask to arguments

        return {
            "command": masked_commands,   # masked commands
            "args": masked_args,          # masked arguments
            "tgt_commands": tgt_commands, # original commands (labels)
            "tgt_args": tgt_args,         # original args (labels)
            "cmd_mask": rand_cmd_mask,    # mask positions for commands
            "arg_mask": rand_arg_mask,    # mask positions for args
            "id": [sample["id"] for sample in batch]  # ID of the samples
        }

    return collate_fn  # Return the collate function



