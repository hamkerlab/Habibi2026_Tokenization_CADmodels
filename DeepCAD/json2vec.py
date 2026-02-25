import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed
import sys
from DeepCAD.cadlib.extrude import CADSequence
from DeepCAD.cadlib.macro import *

class json2vec:
    """
    A class to convert CAD JSON files to vector representations and save them as HDF5 files.

    Attributes:
        base_dir (str): The directory containing the input JSON files.
        output_dir (str): The directory where the output HDF5 files will be saved.
    """

    def __init__(self, base_dir, output_dir):
        """
        Initializes the json2vec class with base and output directories.

        Args:
            base_dir (str): The directory containing the input JSON files.
            output_dir (str): The directory where the output HDF5 files will be saved.
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_one(self, data_id, type):
        """
        Process one CAD json file and save as .h5.

        Args:
            data_id (str): The identifier for the CAD JSON file (without extension).
            type (str): The type of data (train, val, test) for organizing output files.
        """
        json_path = os.path.join(self.base_dir, f"{data_id}.json")

        try:
            with open(json_path, "r") as fp:
                data = json.load(fp)

            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            cad_seq.numericalize()
            cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=True)

        except Exception as e:
            print("failed:", data_id)
            return

        if MAX_TOTAL_LEN < cad_vec.shape[0] or cad_vec is None:
            print("exceed length condition:", data_id, cad_vec.shape[0])
            return

        save_dir = os.path.join(self.output_dir, type)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{data_id}.h5")
        with h5py.File(save_path, "w") as fp:
            fp.create_dataset("vec", data=cad_vec, dtype=int)

    def preprocess(self, split_json, n_jobs=10):
        """
        Preprocess the JSON files by reading a split file and processing each file in parallel.

        Args:
            split_json (str): The path to the JSON file that contains the split data (train, val, test).
            n_jobs (int): The number of parallel jobs to run for processing files.
        """
        with open(split_json, "r") as f:
            split_data = json.load(f)

        for type in ["train", "val", "test"]:
            ids = split_data.get(type, [])

            Parallel(n_jobs=n_jobs, verbose=2)(
                delayed(self.process_one)(data_id, type) for data_id in ids
            )





