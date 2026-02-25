**Training**

This section describes how to train models for each method.

The project supports three training pipelines:

- WordPiece-naive
- WordPiece-DC
- DeepCAD

## Train WordPiece models
For training WordPiece-naive/WordPiece-DC, run this command:

```
python WordPiece/train.py --method wp_dc/wp_naive
```

## Train DeepCAD model
For training DeepCAD, run this command:

```
python DeepCAD/train.py
```