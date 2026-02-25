# Habibi2026_Tokenization_CADmodels

Source code of the article *Influence of Tokenization Strategies on the Prediction of CAD Model Descriptions* by Sayeda Hadisa Habibi, Julia Bergelt, Michael Teichmann, Fred H. Hamker, accepted for the International Symposium on Hybrid Intelligence in Product and Production Engineering (2026).

## Aim 

We compare three different tokenization strategies: 
- a common method from Natural Language Processing (NLP)
- a method inspired by DeepCAD (http://arxiv.org/abs/2105.09492)
- a hybrid approach that combines elements of both

on advanced text-based representations for CAD models.

We base our comparison of the different tokenization methods on the metrics token fertility and prediction performance on masked sequence parts, using a lightweight BERT-style transformer model and the DeepCAD dataset.

Getting Started 
- [Installation](docs/INSTALL.md)

- [Data Preparation](docs/DATASET_PREPARATION.md)

- [Training](docs/TRAIN_EVAL.md)

