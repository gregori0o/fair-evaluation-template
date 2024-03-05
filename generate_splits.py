import json

import numpy as np
from sklearn.model_selection import KFold

from config import K_FOLD
from load_data import DATA_SPLITS_DIR, DatasetName, GraphsDataset
from utils import NpEncoder


def generate(dataset_name: DatasetName):
    path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}.json"
    size = GraphsDataset(dataset_name).size
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=12)
    indexes = np.arange(size)

    folds = []
    for train, test in kfold.split(indexes):
        folds.append({"train": train, "test": test})
    dumped = json.dumps(folds, cls=NpEncoder)
    with open(path, "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    for dataset_name in DatasetName:
        generate(dataset_name)
