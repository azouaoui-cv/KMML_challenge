import os

FILES = {0: {"train_mat": "Xtr0_mat100.csv",
             "train": "Xtr0.csv",
             "test_mat": "Xte0_mat100.csv",
             "test": "Xte0.csv",
             "label": "Ytr0.csv"},
         1: {"train_mat": "Xtr1_mat100.csv",
             "train": "Xtr1.csv",
             "test_mat": "Xte1_mat100.csv",
             "test": "Xte1.csv",
             "label": "Ytr1.csv"},
         2: {"train_mat": "Xtr2_mat100.csv",
             "train": "Xtr2.csv",
             "test_mat": "Xte2_mat100.csv",
             "test": "Xte2.csv",
             "label": "Ytr2.csv"}}

CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, "data")
RESULT_DIR = os.path.join(CWD, "results")
LOGGING_DIR = os.path.join(CWD, "logging")
EMBEDDING_DIR = os.path.join(CWD, "embeddings")
