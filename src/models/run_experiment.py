# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import (BayesianRidge, PassiveAggressiveRegressor,
                                  SGDRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.tools.kfolds import KFoldsExperiment, Model
from src.tools.metrics import calc_pearson, calc_spearman
from src.tools.utils import *

MODELS = {
    'brr': BayesianRidge(),
    'sgdr': SGDRegressor(verbose=1, tol = 1e-6, n_iter_no_change=10, max_iter=1000),
    'ipca-sgdr': Pipeline(steps=[('ipca', IncrementalPCA(n_components=512,batch_size=1024)), ('sgdr', SGDRegressor(verbose=0, tol = 1e-6, n_iter_no_change=10, max_iter=1000))]),
    'par': PassiveAggressiveRegressor(),
    "mpnet": AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-mpnet-base-v2", num_labels=1)
}


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('model_name', type=click.STRING)
@click.argument('out_path', type=click.Path(writable=True))
@click.argument('feat_col', type=click.STRING)
@click.argument('label_col', type=click.STRING)
@click.argument('folds', type=click.INT)
@click.option('--group_by', type=click.STRING, default=None)
@click.option('--sample', type=click.FLOAT, default=None)
@click.option('--bs', type=click.INT, default=None)
@click.option('--kf_type', type=click.STRING, default='simple')
def main(data_path, model_name, out_path, feat_col, label_col, folds, group_by, sample, bs, kf_type):
    """
    Run an experiment to train, fine-tine or evaluate models

    :param data_path: The path to the data.
    :type data_path: str

    :param model_name: The name of the model to train, fine-tune or evaluate.
    :type model_name: str

    :param out_path: The path to the output directory.
    :type out_path: str

    :param feat_col: The name of the feature column.
    :type feat_col: str

    :param label_col: The name of the label column.
    :type label_col: str

    :param folds: The number of folds to use for cross-validation.
    :type folds: int

    :param group_by: The name of the column to group by.
    :type group_by: str

    :param sample: The fraction of the data to use.
    :type sample: float

    :param bs: The batch size to use.
    :type bs: int

    :param kf_type: The type of cross-validation to use.
    :type kf_type: str

    :return: None
    """

    
    logger = logging.getLogger(__name__)
    logger.info('running experiment')
    logger.info(f'data path is: {data_path}')
    logger.info(f'model name is: {model_name}')
    logger.info(f'output path is: {out_path}')
    logger.info(f'feature column is: {feat_col}')
    logger.info(f'label column is: {label_col}')
    logger.info(f'number of folds is: {folds}')
    logger.info(f'group by column is: {group_by}')
    logger.info(f'sample fraction is: {sample}')
    logger.info(f'batch size is: {bs}')
    logger.info(f'cross-validation type is: {kf_type}')


    if not os.path.exists(out_path):
        os.makedirs(out_path)


    # Load data
    try:
        if 'train' in data_path:
            # Load both train and val and concat them
            data1 = load_data(data_path)
            data2 = load_data(data_path.replace('train', 'val')) if os.path.exists(data_path.replace('train', 'val')) else pd.DataFrame()
            data = pd.concat([data1, data2]).reset_index(drop=True)
        elif 'val' in data_path:
            # Load both train and val and concat them
            data1 = load_data(data_path)
            data2 = load_data(data_path.replace('val', 'train'))
            data = pd.concat([data1, data2]).reset_index(drop=True)
        else:
            data = load_data(data_path)
        # Explode data if it contains lists
        if isinstance(data[feat_col].iloc[0], list):
            data = data.explode(feat_col).reset_index(drop=True)
            click.echo("Exploded data.")
        click.echo(f"Loaded {len(data)} rows of data.")
        # Sample data keeping the label distribution (which is continuous)
        if sample is not None:
            # Sort data by label column (continuous)
            data = data.sort_values(label_col)
            # Save one out of every sample rows and discard the rest
            data = data.iloc[::int(1/sample)]
            logger.info(f"Sampled {sample} of the data.")
            logger.info(f"New data size: {len(data)}")
    except ValueError as e:
        logger.error(e)
        return
    
    # Load model
    try:
        model = MODELS[model_name]
        logger.info(f"Loaded model: {model_name}")
    except KeyError as e:
        logger.error(f"Model {model_name} not found.")
        return
    
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2") if model_name == "mpnet" else None
    logger.info(f"Loaded tokenizer: {tok}")
    
    # Prepare data
    # Check if features column contains paths
    data_prepared = pd.DataFrame()
    data_prepared['X'] = data[feat_col]
    data_prepared['y'] = data[label_col]
    data_prepared['groups'] = data[group_by]

    # Trash data to avoid memory issues
    del data

    metrics = [mean_squared_error, calc_pearson, calc_spearman]

    kf = KFoldsExperiment(Model(model,tok,"cuda"), data_prepared, metrics, k=folds, group_by='groups', batch_size=bs, type=kf_type)

    results = kf.run(out_path)
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_path, 'results.csv'), index=False)

    logger.info('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()