import os

import click
import numpy as np
import pandas as pd
from datasets import Dataset
from joblib import dump
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import PreTrainedModel

from src.tools.utils import build_memarray_from_files


class Model:
    """
    Model Class to wrap different sources, APIs and whatnot
    """

    def __init__(self, model, tokenizer=None, device=None):
        """
        Model Class to wrap different sources, APIs and whatnot
        """
        self.model = model
        if isinstance(self.model, PreTrainedModel):
            self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def fit(self, data, save_path, batch_size=None):
        """
        Fit the model
        """
        # Check if X column contains paths
        if isinstance(data['X'].iloc[0], str) and os.path.exists(data['X'].iloc[0]):
            data['X'] = data['X'].apply(lambda x: np.load(x))
            X = np.vstack(data['X'].to_numpy())
            # X = build_memarray_from_files(data['X'].values)
            y = data['y'].to_numpy()
            assert len(X) == len(y), "X and y must have the same length"
        elif isinstance(data['X'].iloc[0], np.ndarray) or isinstance(data['X'].iloc[0], list):
            X = np.vstack(data['X'].to_numpy())
            y = data['y'].to_numpy()
            assert len(X) == len(y), "X and y must have the same length"
        elif isinstance(data['X'].iloc[0], str):
            data.rename(columns={'y': 'labels'}, inplace=True)     
        else:
            raise ValueError("Unsupported X type")
        if batch_size is None:
            if isinstance(self.model, Pipeline):
                # param_grid = {
                #     'ipca__n_components': [128, 256, 512]
                # }
                # search = GridSearchCV(self.model, param_grid, cv=GroupKFold(n_splits=5), verbose=1, n_jobs=-1)
                # search.fit(X, y, groups=data['groups'])
                # click.echo(f"Best params: {search.best_params_}")
                # self.model = search.best_estimator_
                self.model.fit(X, y)
            elif hasattr(self.model, 'fit'):
                # click.echo("!!!!!!!!!!!!!!!!!!!!!!Fitting model line 50")
                self.model.fit(X, y)
            elif isinstance(self.model, PreTrainedModel):
                self.train_model(data, save_path, batch_size=64)
            else:
                raise AttributeError("Model does not have fit attribute")
        else:
            if hasattr(self.model, 'partial_fit'):
                for i in trange(0, len(X), batch_size, desc="Fitting batches"):
                    self.model.partial_fit(X[i:i+batch_size], y[i:i+batch_size])
            elif isinstance(self.model, PreTrainedModel):
                self.train_model(data, save_path, batch_size=batch_size)
            else:
                raise AttributeError("Model does not have partial_fit attribute")
            
    def train_model(self, data, save_path, batch_size=64, lr=1e-5):
        """
        Train the model
        """
        dataset = Dataset.from_pandas(data).map(lambda x: self.tokenizer(x['X'], truncation=True, padding='max_length'), batched=True)
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = Adam(self.model.parameters(), lr=lr)
        
        patience_left = 3
        best_loss = np.inf

        for epoch in trange(3, desc="Training epochs"):
            click.echo(f"Running epoch {epoch}")
            train_loss = self._run_epoch(dataloader, optimizer)
            if train_loss < best_loss:
                best_loss = train_loss
                click.echo(f"Train loss improved. Saving model.")
                self.model.save_pretrained(save_path)
                patience_left = 3
            else:
                click.echo(f"Train loss did not improve. Patience left: {patience_left}")
                patience_left -= 1
            if patience_left == 0:
                click.echo("Patience exhausted. Stopping training.")
                break
    
    def _run_epoch(self, dataloader, optimizer, train=True):
        """
        Run an epoch
        """
        self.model.train() if train else self.model.eval()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Running epoch"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return total_loss / len(dataloader)


    def fit_tokenizer(self, data, save_path=None):
        """
        Fit the tokenizer
        """
        corpus = (data['X'][i : i + 1000] for i in range(0, len(data['X']), 1000))
        new_tokenizer = self.tokenizer.train_new_from_iterator(corpus,vocab_size=32000)
        if save_path:
            new_tokenizer.save_pretrained(save_path)
        self.tokenizer = new_tokenizer

    def predict(self, data):
        """
        Predict
        """
        if isinstance(data['X'].iloc[0], str) and os.path.exists(data['X'].iloc[0]):
            data['X'] = data['X'].apply(lambda x: np.load(x))
            X = np.vstack(data['X'].to_numpy())
            # X = build_memarray_from_files(data['X'].values)
        elif isinstance(data['X'].iloc[0], np.ndarray) or isinstance(data['X'].iloc[0], list):
            X = np.vstack(data['X'].to_numpy())
        elif isinstance(data['X'].iloc[0], str):
            data['text'] = data['X']
        else:
            raise ValueError("Unsupported X type")
        assert len(X) == len(data['y']), "X and y must have the same length"

        y = data['y'].to_numpy()
        if hasattr(self.model, 'predict'):
            y_hat = self.model.predict(X)
        elif isinstance(self.model, PreTrainedModel):
            dataset = Dataset.from_pandas(data).map(lambda x: self.tokenizer(x['X'], truncation=True, padding='max_length'), batched=True)
            dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            y_hat = []
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                y_hat.extend(outputs.logits.argmax(dim=1).cpu().numpy())
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
            results[metric_name] = metric(data['y'].to_numpy(), y_hat)
        return results

    def save(self, path):
        """
        Save
        """
        dump(self.model, path)