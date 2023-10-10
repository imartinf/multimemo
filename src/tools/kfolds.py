import os
from typing import Any, Iterator

import click
import numpy as np
from joblib import dump
from numpy.lib.format import open_memmap
import pandas

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GroupKFold, KFold, PredefinedSplit, StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from src.models.model import Model

def compute_test_fold_indices(data: pandas.DataFrame, n_splits: int, random_state: int | None = 42) -> np.ndarray:
    """
    Compute test indices in a stratified way.
    The data should be splitted in n_splits in a way that preserves the groups AND the continuous label distribution (data['y']).
    """
    # Random state for reproducibility
    np.random.seed(random_state)
    # Order the dataframe by y and then groups
    data = data.sort_values(by=['y', 'groups'])
    # Create a 'fold' column. It should assign the same fold to each of the samples that belong to the same group
    # Start from 0 to n_splits-1 and over again
    unique_groups = data['groups'].unique()
    fold_labels = np.arange(n_splits)
    fold_dict = {}
    for _, group in enumerate(unique_groups):
        # Assign a random element from fold_labels and remove it from the list
        fold = np.random.choice(fold_labels)
        fold_labels = np.delete(fold_labels, np.argwhere(fold_labels == fold))
        fold_dict[group] = fold
        if len(fold_labels) == 0:
            fold_labels = np.arange(n_splits)
    data['fold'] = data['groups'].map(fold_dict)
    data.sort_index(inplace=True)
    return data['fold'].to_numpy()


class KFoldsExperiment:
    """
    K Folds Experiment Class
    """

    def __init__(self, model: Model, data, metrics=None, k=5, shuffle=True, type='simple', group_by=None, random_state=42, batch_size=None, fit_tokenizer=False):
        """
        K Folds Experiment Class
        """
        self.model = model
        self.data = data
        self.metrics = metrics
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        self.type = type
        self.group_by = group_by
        self.batch_size = batch_size
        self.fit_tokenizer = fit_tokenizer

    def run(self, out_path):
        """
        Run the experiment
        """
        results = {}
        if self.type == 'simple':
            kf = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        elif self.type == 'group':
            kf = GroupKFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        elif self.type == 'stratified':
            kf = StratifiedKFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        elif self.type == 'stratified_group':
            if type_of_target(self.data['y']) == 'continuous':
                kf = PredefinedSplit(compute_test_fold_indices(self.data, self.k, self.random_state))
            else:
                kf = StratifiedGroupKFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        else:
            raise ValueError(f"Invalid type: {self.type}")

        for fold, (train_index, test_index) in enumerate(kf.split(self.data, groups=self.data[self.group_by] if self.group_by is not None else None)):
            click.echo(f"Running fold {fold}")
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            if self.fit_tokenizer:
                click.echo("Fitting tokenizer")
                self.model.fit_tokenizer(train_data, save_path=os.path.join(out_path, f"tokenizer_fold_{fold}"))
            click.echo(f"Training on {len(train_data)} samples")
            self.model.fit(train_data, batch_size=self.batch_size, save_path=os.path.join(out_path, f"model_fold_{fold}"))
            click.echo(f"Evaluating on {len(test_data)} samples")
            fold_res = self.model.evaluate(test_data, metrics=self.metrics)
            click.echo(f"Results for fold {fold}: {fold_res}")
            self.model.save(os.path.join(out_path, f"model_fold_{fold}.joblib"))
            results[f"fold_{fold}"] = fold_res
        return results