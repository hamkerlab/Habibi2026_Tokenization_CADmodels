from DeepCAD.json2vec import json2vec
from WordPiece.WordPiece_naive.json2txt import json2txt
from WordPiece.WordPiece_DC.CADprocessor import CADProcessor
from config.cfg_data import PATHS

import argparse
from pathlib import Path

def prepare_wp_naive(cfg, split_path: Path | None = None):
    
    processor = json2txt(PATHS.cad_json, PATHS.wp_naive)
    processor.normalize(str(PATHS.splits))

def prepare_wp_dc(cfg, split_path: Path | None = None):

    processor = CADProcessor(PATHS.cad_json, PATHS.wp_dc)
    processor.normalize(str(PATHS.splits))

def prepare_deepcad(cfg, split_path: Path | None = None):
    
    processor = json2vec(PATHS.cad_json, PATHS.deepcad)
    processor.preprocess(str(PATHS.splits))

def main():
    parser = argparse.ArgumentParser("Prepare CAD data")

    parser.add_argument("--wp_naive", action="store_true", help="Run WordPiece-naive preprocessing")
    parser.add_argument("--wp_dc", action="store_true", help="Run WordPiece-DC preprocessing")
    parser.add_argument("--deepcad", action="store_true", help="Run DeepCAD preprocessing")
    parser.add_argument("--split", type=Path ,default=None)

    args = parser.parse_args()

    if args.deepcad:
        prepare_deepcad(args.split)
    elif args.wp_naive:
        prepare_wp_naive(args.split)
    else:
        prepare_wp_dc(args.split)

if __name__ == "__main__":
    main()


