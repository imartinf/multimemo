# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.tools.text_processing import apply_similarity_filter

from src.tools.utils import *


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
def main(data_path, config_path):
    """
    Script to train or fine-tune models

    :param data_path: The path to the data.
    :type data_path: str

    :param config_path: The path to the config file.
    :type config_path: str

    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info('training models')
    logger.info(f'data path is: {data_path}')
    logger.info(f'config path is: {config_path}')

    # Load data
    data = explode_data(load_data(data_path),['captions', 'recaptions'])
    for split, df in data.items():
        logger.info(f'{split} data shape: {df.shape}')


    # Load config
    config = load_config(config_path)
    logger.info(f'config: {config}')

    # Filter data if specified in config
    if config['sim_filter']:
        logger.info('Filtering data')
        data = data_exploded = apply_similarity_filter(data, tokenizer, sentence_model, 'captions', 'recaptions', 'sim', 'recaptions_filtered', config['sim_filter_strategy'], config['sim_filter_threshold'])
        for split, df in data.items():
            logger.info(f'{split} data shape: {df.shape}')

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