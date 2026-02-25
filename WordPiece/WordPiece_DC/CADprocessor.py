class CADProcessor:
    """
    A class to process CAD data for machine learning applications.

    Attributes:
    ----------
    input_folder : str
        Path to the folder containing input JSON files.
    output_folder : str
        Path to the folder where output files will be stored.
    num_regex : re.Pattern
        Regular expression pattern to identify numbers in text.

    Methods:
    -------
    normalize_floats(item):
        Normalize numbers to 4 decimal places.
    
    process_curve(curve):
        Convert a curve object (Circle, Line, Arc) into a list representation.
    
    process_extrude(extrude):
        Convert an Extrude object into a list representation.
    
    process_json(json_path):
        Process a single JSON file and return a normalized text representation.
    
    scale(files):
        Compute normalization scale (min/max) from the training set.
    
    _normalize(text, min_val, max_val):
        Normalize the text based on the provided min and max values.
    
    normalize(json_path, n_jobs=10):
        Normalize the data in the specified JSON file and save to output folder.
    
    build_corpus():
        Build a corpus.txt file from all normalized text files.
    """
import sys, os
sys.path.append("/scratch/hsay/Transformer/cad-transformer")
from DeepCAD.cadlib.extrude import CADSequence, Extrude
import numpy as np
from DeepCAD.cadlib.macro import *
import json
import re
from joblib import Parallel, delayed
from pathlib import Path

class CADProcessor:
    def __init__(self, input_folder, output_folder):
        """
        Initialize CADProcessor with input_folder containing train/, val/, test/
        Output will be stored under input_folder/WordPiece-DC/
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.num_regex = re.compile(r'(?<![A-Za-z0-9_+\-/])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_+\-/])')

    @staticmethod
    def normalize_floats(item):
        """
        Normalize numbers
        - Round floats to 4 decimal places
        """
        if isinstance(item, (np.floating, float)):
            return round(float(item), 4)
        elif isinstance(item, (np.integer, int)):
            return int(item)
        elif isinstance(item, (list, tuple)):
            return [CADProcessor.normalize_floats(x) for x in item]
        elif isinstance(item, np.ndarray):
            return [CADProcessor.normalize_floats(x) for x in item.tolist()]
        else:
            return item
            
    def process_curve(self, curve):
        """
        Convert a curve object (Circle, Line, Arc) into a list representation
        Returns None for unsupported curve types
        """

        curve_type = curve.__class__.__name__

        if curve_type == "Circle":
            # Circle: [type, center_x, center_y, radius]
            return self.normalize_floats([curve_type,
                        curve.center[0],
                        curve.center[1],
                        curve.radius
            ])

        elif curve_type == "Line":
            # Line: [type, end_point_x, end_point_y]
            return self.normalize_floats([curve_type,
                        curve.end_point[0],
                        curve.end_point[1]
            ])
            
        elif curve_type == "Arc":
            # Arc: [type, end_point_x, end_point_y, sweep_angle, clockwise_flag]
            sweep_angle = max(abs(curve.start_angle - curve.end_angle), 1)
            return self.normalize_floats([curve_type,
                        curve.end_point[0],
                        curve.end_point[1],
                        sweep_angle,
                        int(curve.clock_sign)
            ])
        else:
            return None
            
    def process_extrude(self, extrude):
        """
        Convert an Extrude object into a list of representation
        Processes all curves in the profile and then adds extrude operation info
        """
        result = []

        profile = extrude.profile
        # Iterate through all loops and curves in the profile
        for loop in profile.children:
            for curve in loop.children:
                curve_data = self.process_curve(curve)
                if curve_data:
                    result.append(curve_data)

        # Add extrude operation information
        result.append(self.normalize_floats([
            "Extrude",
            *extrude.sketch_plane.to_vector()[3:], # Extract plane vector
            extrude.sketch_pos[0],                 # Sketch position X
            extrude.sketch_pos[1],                 # Sketch position Y
            extrude.sketch_pos[2],                 # Sketch position Z
            extrude.sketch_size,                   # Sketch size
            extrude.extent_one,                    # Extrude extent one
            extrude.extent_two,                    # Extrude extent two
            extrude.operation,                     # Operation type
            extrude.extent_type                    # Extent type
        ]))
        return result
 
    def process_json(self, json_path):
        """
        Process a single JSON file
        - Load JSON CAD data
        - Convert ExtrudeFeatures to CADSequence 
        - Normalize and process all extrudes 
        - Save results to a txt file
        """

        with open(json_path, "r") as f:
            data = json.load(f)

            # Build CAD sequence from JSON data
            seq = []
            for item in data.get("sequence", []):
                if item["type"] == "ExtrudeFeature":
                    seq.extend(Extrude.from_dict_DC(data, item["entity"]))
            
            # Build bounding box
            bbox_info = data["properties"]["bounding_box"]
            max_point = np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
            min_point = np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
            bbox = np.stack([max_point, min_point], axis=0)
            
            # Create CAD sequence object
            cad_seq = CADSequence(seq, bbox)

            # Process all extrudes in the sequence
            lines = []
            for extrude in cad_seq.seq:
                lines.extend(self.process_extrude(extrude))

            text = "\n".join(" ".join(map(str, row)) for row in lines)
            return text

    def scale(self, files):
        """
        Compute normalization scale (min/max) from trainin set
        """
        all_numbers = []
        for file_path in files:
            text = self.process_json(file_path)
            nums = [float(x) for x in self.num_regex.findall(text)]
            all_numbers.extend(nums)

        min_val = np.percentile(all_numbers, 1)
        max_val = np.percentile(all_numbers, 99)

        print(f"min={min_val:.4f}, max={max_val:.4f}")

        return min_val, max_val

    def _normalize(self, text, min_val, max_val):

        def normalize_match(match):
            val = float(match.group())
            norm_val = 2 * (val - min_val) / (max_val - min_val) - 1
            return str(round(max(-1, min(1, norm_val)), 4))

        return self.num_regex.sub(normalize_match, text)
    
    def normalize(self, json_path, n_jobs=10):
        with open(json_path, "r") as f:
            split_data = json.load(f)

        json_filename = os.path.basename(json_path)

        dataset = {}
        for subset, ids in split_data.items():
            out_dir = self.output_folder / subset
            out_dir.mkdir(parents=True, exist_ok=True)

            dataset[subset] = []
            for file_id in ids:
                json_file = self.input_folder / f"{file_id}.json"

                if os.path.basename(json_file) == json_filename:
                    continue

                dataset[subset].append(json_file)

        min_val, max_val = self.scale(dataset["train"])

        print(f"Normalization scale: min={min_val}, max={max_val}")

        for subset, files in dataset.items():
            out_dir = self.output_folder / subset
            out_dir.mkdir(parents=True, exist_ok=True)

            def process_file(file_path):
                try:
                    text = self.process_json(file_path)
                    normalized_text = self._normalize(text, min_val, max_val)
                    out_path = out_dir / (file_path.stem + ".txt")
                    with open(out_path, "w") as f:
                        f.write(normalized_text + "\n")

                except Exception as e:
                    print(f"Error in {file_path}: {e}")

            Parallel(n_jobs=n_jobs, verbose=2)(
                delayed(process_file)(file_path) for file_path in files
            )

        self.build_corpus()

    def build_corpus(self):
        """
        Build corpus.txt from all normalized text files
        """
        corpus_path = self.output_folder / "corpus.txt"
        with open(corpus_path, "w") as corpus:
            for split in ["train", "val", "test"]:
                split_dir = self.output_folder / split
                for fname in sorted(os.listdir(split_dir)):
                    if fname.endswith(".txt"):
                        with open(os.path.join(split_dir, fname), "r") as f:
                            corpus.write(f.read().strip() + "\n")
        print(f"Corpus built at {corpus_path}")



