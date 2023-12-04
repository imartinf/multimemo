import os
from typing import Any, Iterator

import click
import numpy as np
import pandas
import pandas as pd
from joblib import dump
from numpy.lib.format import open_memmap
from sklearn.model_selection import (GroupKFold, KFold, PredefinedSplit,
                                     StratifiedGroupKFold, StratifiedKFold)
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm
import wandb

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
    # Log mean and std y value for each fold
    for fold in range(n_splits):
        fold_data = data[data['fold'] == fold]
        click.echo(f"Fold {fold} - Mean y: {fold_data['y'].mean()} - Std y: {fold_data['y'].std()} | Num samples: {len(fold_data)}")
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

    def run(self, out_path, **kwargs):
        """
        Run the experiment
        """
        feat_col = kwargs.get('feat_col', 'X')
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
            if isinstance(feat_col,list) and len(feat_col) > 1:
                click.echo("Applying data augmentation")
                # Each value of feat_col is the name of a column that contains (augmented) data
                # We need to stack them together with their corresponding labels
                # Vertically concatenate each column, repeating the labels
                # Each column must have the same number of rows

                aug_train_data = pd.DataFrame(columns=['X','y','groups'])
                filter_thres = kwargs.get('filter_thres', None)
                filter_col = kwargs.get('filter_col', None)
                for i, col in enumerate(feat_col):
                    # Vertically concatenate each column, repeating the labels
                    # Each column must have the same number of rows
                    aux = train_data[[col,'y','groups', filter_col]].rename(columns={col:'X'}) if filter_col is not None else train_data[[col,'y','groups']].rename(columns={col:'X'})
                    # Remove the rows that have a value below filter_thres in filter_col if specified
                    if filter_col is not None and i > 0:
                        aux = aux[aux[filter_col] >= filter_thres]
                        aug_train_data = pd.concat([aug_train_data, aux.drop(columns=[filter_col])], axis=0, ignore_index=True)
                    else:
                        aug_train_data = pd.concat([aug_train_data, aux], axis=0, ignore_index=True)
                    click.echo(f"Added {len(aux)} rows from {col}")
                train_data = aug_train_data
                wandb.log({"num_train_samples": len(train_data)})
                test_data['X'] = test_data[feat_col[0]]

            else:
                train_data['X'] = train_data[feat_col]
                test_data['X'] = test_data[feat_col]
            if self.fit_tokenizer:
                click.echo("Fitting tokenizer")
                self.model.fit_tokenizer(train_data, save_path=os.path.join(out_path, f"tokenizer_fold_{fold}"))
            click.echo(f"Training on {len(train_data)} samples")
            self.model.fit(train_data, batch_size=self.batch_size, save_path=os.path.join(out_path, f"model_fold_{fold}"))
            click.echo(f"Evaluating on {len(test_data)} samples")
            fold_res, y_hat = self.model.evaluate(test_data, metrics=self.metrics)
            click.echo(f"Results for fold {fold}: {fold_res}")
            # # Plot outlier candidates on wandb
            # wandb.sklearn.plot_outlier_candidates(
            #     self.model.model,
            #     np.vstack(train_data['X'].to_numpy()),
            #     train_data['y'].to_numpy()
            # )
            # # Plot residuals
            # wandb.sklearn.plot_residuals(
            #     self.model.model,
            #     np.vstack(test_data['X'].to_numpy()),
            #     test_data['y'].to_numpy()
            # )
            wandb.log(fold_res)
            self.model.save(os.path.join(out_path, f"model_fold_{fold}.joblib"))
            results[f"fold_{fold}"] = fold_res
        mean_spearman = np.mean([results[fold]['calc_spearman'] for fold in results.keys()])
        click.echo(f"Mean spearman: {mean_spearman}")
        wandb.summary["mean_spearman"] = mean_spearman
        return results
