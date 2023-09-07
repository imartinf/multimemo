# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from sklearn.metrics import mean_squared_error
from src.tools.kfolds import KFoldsExperiment, Model
from src.tools.metrics import calc_pearson, calc_spearman

from src.tools.utils import *

from sklearn.linear_model import BayesianRidge


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('model_name', type=click.STRING)
@click.argument('out_path', type=click.Path(writable=True))
@click.argument('feat_col', type=click.STRING)
@click.argument('label_col', type=click.STRING)
@click.argument('folds', type=click.INT)
@click.option('--group_by', type=click.STRING, default=None)
def main(data_path, model_name, out_path, feat_col, label_col, folds, group_by):
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

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    # Load data
    try:
        data = load_data(data_path)
        data = data.sample(frac=0.1).reset_index(drop=True)
    except ValueError as e:
        logger.error(e)
        return
    
    # Load model
    if model_name == "brr":
        model = BayesianRidge()
    else:
        logger.error(f"Model {model_name} not found.")
        return
    
    # Prepare data
    # Check if features column contains paths
    data_prepared = pd.DataFrame()
    data_prepared['X'] = data[feat_col]
    data_prepared['y'] = data[label_col]
    data_prepared['groups'] = data[group_by]

    # Trash data to avoid memory issues
    del data

    metrics = [mean_squared_error, calc_pearson, calc_spearman]

    kf = KFoldsExperiment(Model(model), data_prepared, metrics, k=folds, group_by='groups')

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