# -*- coding: utf-8 -*-
import json
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from tqdm import tqdm

from src.tools.chatgpt import *
from src.tools.prompts import PROMPTS
from src.tools.utils import save_json


@click.command()
@click.argument('config_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('-temperature', default=0.9, help='The temperature to use for the response generation.')
def main(config_filepath, input_filepath, output_path, temperature):
    """ 
    Script to automatically generate responses using ChatGPT using a dataset of original captions and a prompt.

    :param config_filepath: The path to the configuration file.
    :type config_filepath: str

    :param input_filepath: The path to the input data.
    :type input_filepath: str

    :param output_path: The path to the output data.
    :type output_filepath: str

    :param temperature: The temperature to use for the response generation.
    :type temperature: float

    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating responses dataset')
    tqdm.pandas()

    # Get chatgpt config from environment and set up the ChatGPT object
    gptconfig = {
        'MODEL': os.environ.get('MODEL'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
        'OPENAI_API_VERSION': os.environ.get('OPENAI_API_VERSION')
    }

    gptconfig['TEMPERATURE'] = temperature
    base_prompt = PROMPTS['mem_exp']

    chatgpt = ChatGPT(gptconfig,logger)

    # Get split from filename
    split = os.path.basename(input_filepath).split('_')[1]

    data = pd.read_json(input_filepath, orient='records', lines=True)
    data['actions'] = data['action_labels'].progress_apply(lambda x: ' '.join(x).replace('+', ' '), desc='Processing action keywords')

    with open(config_filepath, 'r') as f:
        config = json.load(f)

    data['responses'] = ''

    for i, row in tqdm(data.iterrows(), total=len(data), desc='Generating responses'):
        responses = chatgpt.get_responses(row, base_prompt)
        # Save response in json file
        for k, response in responses:
            try:
                save_json(response, os.path.join(output_path, config['EXP_NAME'], f"{row['filename']}_{k}.json"), 4)
            except Exception as e:
                logger.error(f"Error saving response of {row['filename']}_{k}.json: {e}")
        data.at[i, 'responses'] = responses

    data.to_json(os.path.join(output_path, f"memento_{split}_data_{config['EXP_NAME']}.json"))

    # Second pass for solving errors
    for i, row in tqdm(data.iterrows(), total=len(data), desc='Generating responses (2nd pass)'):
        responses = chatgpt.retry_get_responses_if_error_in_response(row, chatgpt, logger, base_prompt)
        # Save response in json file
        for k, response in responses:
            try:
                save_json(response, os.path.join(output_path, config['EXP_NAME'], f"{row['filename']}_{k}.json"), 4)
            except Exception as e:
                logger.error(f"Error saving response of {row['filename']}_{k}.json: {e}")
        data.at[i, 'responses'] = responses

    data.to_json(os.path.join(output_path, f"memento_{split}_data_{config['EXP_NAME']}.json"))
    logger.info("Extracting texts...")
    data['answer'] = data['responses'].progress_apply(lambda responses: get_text_from_responses(responses))
    # logger.info("Extracting scores...")
    # train['score_preds'] = train['responses'].progress_apply(lambda responses: get_score_from_responses(responses))
    logger.info("Computing tokens...")
    data['total_used_tokens'] = data['responses'].progress_apply(lambda responses: get_tokens_from_responses(responses))

    logger.info("Saving...")
    data.to_json(os.path.join(output_path, f"memento_{split}_data_{config['EXP_NAME']}.json"))

    logger.info("Done!")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
