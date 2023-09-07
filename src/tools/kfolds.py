import os
import click
from joblib import dump
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

class Model:
    """
    Model Class to wrap different sources, APIs and whatnot
    """

    def __init__(self, model, tokenizer=None, device=None):
        """
        Model Class to wrap different sources, APIs and whatnot
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def fit(self, data):
        """
        Fit the model
        """
        # Check if X column contains paths
        if os.path.exists(data['X'].iloc[0]):
            data['X'] = data['X'].apply(lambda x: np.load(x))
        X = np.vstack(data['X'].to_numpy())
        y = data['y'].to_numpy()
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        else:
            raise AttributeError("Model does not have fit attribute")

    def predict(self, data):
        """
        Predict
        """
        if os.path.exists(data['X'].iloc[0]):
            data['X'] = data['X'].apply(lambda x: np.load(x))
        X = np.vstack(data['X'].to_numpy())
        y = data['y'].to_numpy()
        if hasattr(self.model, 'predict'):
            y_hat = self.model.predict(X)
        else:
            raise AttributeError("Model does not have predict attribute")
        return y_hat

    def evaluate(self, data, metrics=None):
        """
        Evaluate
        """
        results = {}
        y_hat = self.predict(data)
        if metrics is None:
            metrics = []
        for metric in metrics:
            # Extract metric name from function name
            metric_name = metric.__name__
            results[metric_name] = metric(data['y'], y_hat)
        return results

    def save(self, path):
        """
        Save
        """
        dump(self.model, path)
        

class KFoldsExperiment:
    """
    K Folds Experiment Class
    """

    def __init__(self, model: Model, data, metrics=None, k=5, shuffle=True, group_by=None, random_state=42):
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
            click.echo(f"Training on {len(train_data)} samples")
            self.model.fit(train_data)
            click.echo(f"Evaluating on {len(test_data)} samples")
            fold_res = self.model.evaluate(test_data, metrics=self.metrics)
            click.echo(f"Results for fold {fold}: {fold_res}")
            self.model.save(os.path.join(out_path, f"model_fold_{fold}.joblib"))
            results[f"fold_{fold}"] = fold_res
        return results