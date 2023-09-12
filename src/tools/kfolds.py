import os

import click
import numpy as np
from joblib import dump
from numpy.lib.format import open_memmap
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm

from src.models.model import Model

class KFoldsExperiment:
    """
    K Folds Experiment Class
    """

    def __init__(self, model: Model, data, metrics=None, k=5, shuffle=True, group_by=None, random_state=42, batch_size=None, fit_tokenizer=False):
        """
        K Folds Experiment Class
        """
        self.model = model
        self.data = data
        self.metrics = metrics
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_by = group_by
        self.batch_size = batch_size
        self.fit_tokenizer = fit_tokenizer

    def run(self, out_path):
        """
        Run the experiment
        """
        results = {}
        kf = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state) if self.group_by is None else GroupKFold(
            n_splits=self.k)
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