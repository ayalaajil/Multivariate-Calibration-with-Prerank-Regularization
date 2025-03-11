import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import typing as ty
from sklearn.model_selection import train_test_split
import torch
import time
from mlp import Standalone_RealMLP_TD_S_Classifier, Standalone_RealMLP_TD_S_Regressor
np.random.seed(1)
torch.manual_seed(1)

data_name = "allrep"
data_path = "../data/data/"

def load_json(path):
    return json.loads(Path(path).read_text())

def dataname_to_numpy(dataset_name, dataset_path):

    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    dir_ = Path(os.path.join(dataset_path, dataset_name))

    def load(item):
        return {
            x: ty.cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle = True))  
            for x in ['train', 'val', 'test']
        }

    return (
        load('N') if dir_.joinpath('N_train.npy').exists() else None,
        load('C') if dir_.joinpath('C_train.npy').exists() else None,
        load('y'),
        load_json(dir_ / 'info.json'),
    )

def get_dataset(dataset_name, dataset_path):
    """
    Load the dataset from the numpy files.

    :param dataset_name: str
    :param dataset_path: str
    :return: Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]
    """
    N, C, y, info = dataname_to_numpy(dataset_name, dataset_path)
    N_trainval = None if N is None else {key: N[key] for key in ["train", "val"]} if "train" in N and "val" in N else None
    N_test = None if N is None else {key: N[key] for key in ["test"]} if "test" in N else None

    C_trainval = None if C is None else {key: C[key] for key in ["train", "val"]} if "train" in C and "val" in C else None
    C_test = None if C is None else {key: C[key] for key in ["test"]} if "test" in C else None

    y_trainval = {key: y[key] for key in ["train", "val"]}
    y_test = {key: y[key] for key in ["test"]} 
    
    # tune hyper-parameters
    train_val_data = (N_trainval,C_trainval,y_trainval)
    test_data = (N_test,C_test,y_test)
    return train_val_data,test_data,info

train_val_data, test_data, info = get_dataset(data_name, data_path)
num_features, cat_features, labels = train_val_data
# Concatenate numerical and categorical features column-wise
num_features = pd.DataFrame(np.concatenate((num_features["train"], num_features["val"]), axis=0))
cat_features = pd.DataFrame(np.concatenate((cat_features["train"], cat_features["val"]), axis=0))
X = pd.concat([num_features, cat_features], axis=1)
# Concatenate train and val into a single dataset
# X = np.concatenate((X_train, X_val), axis=0)
y = np.concatenate((labels["train"], labels["val"]), axis=0)

n_valid = 604
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_valid, random_state=42)

classification = True
loss_type = 'rps'
n_repeats = 1
preds = []
seed_offset=1000

# start_time = time.time()
for i in range(n_repeats):
    print(f'Round {i}')
    seed = i + seed_offset
    np.random.seed(seed)
    torch.manual_seed(seed)
    if classification:
        mlp = Standalone_RealMLP_TD_S_Classifier(loss_type=loss_type)
    else:
        mlp = Standalone_RealMLP_TD_S_Regressor()
    mlp.fit(X_train, y_train, X_val, y_val)
    # if classification:
    #     preds.append(mlp.predict_proba(X_test)[:, 1])
    # else:
    #     preds.append(mlp.predict(X_test))
    # print(f'{preds[-1].shape=}')
