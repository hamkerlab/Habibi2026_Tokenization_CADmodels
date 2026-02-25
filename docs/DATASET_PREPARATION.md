# CAD dataset preparation

This document describes the preprocessing and tokenization workflow used in this project.

The data directory can be defined in config/cfg_data.py.

**Step 1**: Run the command below to split the CAD JSON files into training, validation, and test sets.

The command creates a JSON file that contains the ID numbers of each JSON file belonging to each category (training, validation, or test set).

```
python split_dataset.py
```

Output structure:
train_val_test_split.json

**Step 2**: Data preparation for tokenization
Before tokenization, we need to preprocess the dataset for each method

Supported methods:

WordPiece-naive
WordPiece-DC
DeepCAD

**Step 2.1**: Preprocess for WordPiece-naive

Run the following command:

```
python prepare_data.py --wp_naive
```

Output structure:
After running the command, you will find a directory like this:
```
WordPiece-naive
├── train          # .txt files for training 
├── validation     # .txt files for validation
└── test           # .txt files for testing
└── corpus.txt     # for training the WordPiece tokenizer
```

**Step 2.2**: Preprocess for WordPiece-DC
Run the following command:

```
python prepare_data.py --wp_dc
```

Output structure:
After running the command, you will find a directory like this:

```
WordPiece-DC/
├── train/          # .txt files for training
├── validation/     # .txt files for validation
└── test/           # .txt files for testing
└── corpus.txt      # for training the WordPiece tokenizer
```

**Step 2.3**: Preprocess for DeepCAD
Run the following command:

```
python prepare_data --deepcad
```

Output structure:
After running the command, you will find a directory like this:

```
DeepCAD/
├── train/          # .h5 files for training
├── validation/     # .h5 files for validation
└── test/           # .h5 files for testing
```

**Step 3**: Train a WordPiece tokenizer 
Train a tokenizer for WordPiece-naive or WordPiece-DC using the corpus generated in Step 2.1 and 2.2

Command:

```
python WordPiece/train_WPtokenizer.py --method wp_naive/wp_dc --vocab_size 5000

```

**Step 4**: Tokenization of WordPiece-naive and WordPiece-DC
Convert the .txt files in your train/validation/test folders into token IDs using the trained tokenizer

Command:

```
python WordPiece/tokenization.py --method wp_naive/wp_dc --max_len 1024
```

Expected output:

The tokenized tensors will be saved in tokenized_data folders under each train, val and test folder:

```
WordPiece-DC/tokenized_data/
├── train/
│   ├── 00220724.pt
│   ├── 00220726.pt
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```
