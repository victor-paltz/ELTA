# ELTA

## Description

This project is part of a school project, we participate to the Rakuten challenge.
This challenge focuses on the topic of large-scale product type code multimodal (text and image) classification where the goal is to predict each product’s type code as defined in the catalog of Rakuten France.


## Installation

You should use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies.

```bash
pip install -r requirements.txt
```
Our code is written in Python 3.7.

## Usage

First, you should create a folder "data" at the root of the repository, then, put the files "X_train_update.csv" and "Y_train_CVw08PX.csv" inside.

Finally, run the main.py file.

```bash
python3 main.py
```

It is also possible to open the models.ipynb file with jupyter notebook and to run it.

## Files

```
├── code_to_submit.py           -> Script file evaluated by the teachers
├── data                        -> Contains the files X_train_update.csv and Y_train_CVw08PX
│   ├── X_train_update.csv      -> You should add this file
│   ├── Y_train_CVw08PX.csv     -> You should add that file
├── debuging                    -> Folder that contains debuging tools
│   └── info_decorator.py       -> Decorator to print information about execution time and debuging
├── main.py                     -> File to launch in order to run the whole pipeline
├── models.ipynb                -> Notebook that does the same as main.py + hyperparameter fine-tuning
├── preprocessing               -> Folder that gather all the preprocessing functions
│   ├── __init__.py             
│   ├── cleaning_functions.py   -> functions used to clean a sentence
│   ├── preprocessing.py        -> functions to prepare the dataset
│   └── test_cleaning.py        -> test to check that cleaning is working
├── requirements.txt            -> required modules to run the project
└── training
    └── test_models.py          -> contains function to tune hyper-parameters

```

## Tests

In order to test the function to check if nothing is broken, run the following command at the root of the repository:

```bash
pytest
```