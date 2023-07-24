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
from src.tools.utils import *


@click.command()
@click.argument('exp_name', type=click.STRING)
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
@click.argument('prompt_name', type=click.STRING)
@click.option('-temperature', default=0.9, help='The temperature to use for the response generation.')
def main(exp_name, input_filepath, output_path, prompt_name, temperature):
    """ 
    Script to automatically generate responses using ChatGPT using a dataset of original captions and a prompt.

    :param exp_name: The name of the experiment.
    :type exp_name: str

    :param input_filepath: The path to the input data.
    :type input_filepath: str

    :param output_path: The path to the output data.
    :type output_filepath: str

    :param prompt: The keyword of the prompt used for experimentation, as stated in prompts.py.
    :type prompt: str

    :param temperature: The temperature to use for the response generation.
    :type temperature: float

    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating responses dataset')
    tqdm.pandas(desc='Progress')
    logger.info(f'Input filepath is: {input_filepath}')
    logger.info(f'Output path is: {output_path}')
    logger.info(f'Prompt name is: {prompt_name}')
    logger.info(f'Temperature is: {temperature}')
    logger.info(f'Experiment name is: {exp_name}')

    # Get chatgpt config from environment and set up the ChatGPT object
    gptconfig = {
        'MODEL': os.environ.get('MODEL'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
        'OPENAI_API_VERSION': os.environ.get('OPENAI_API_VERSION')
    }

    gptconfig['TEMPERATURE'] = temperature
    base_prompt = PROMPTS[prompt_name]

    chatgpt = ChatGPT(gptconfig,logger)

    # Get split from filename
    split = os.path.basename(input_filepath).split('_')[1]

    logger.info("Reading data from %s" % input_filepath)
    data = pd.read_json(input_filepath, orient='records')
    logger.info("Loaded %s rows of data" % len(data))
    tqdm.pandas(desc='Processing action keywords')
    data['actions'] = data['action_labels'].progress_apply(lambda x: ' '.join(x).replace('+', ' '))
    # Subset for debugging
    data = data.sample(3, random_state=42)

    data['responses'] = ''
    data['total_used_tokens'] = 0
    data['recaptions'] = ''

    # Create output directory if it doesn't exist
    create_dir(os.path.join(output_path, exp_name))

    for i, row in tqdm(data.iterrows(), total=len(data), desc='Generating responses'):
        responses = chatgpt.get_responses(row, base_prompt)
        # logger.info(f"Responses: {responses}")
        # logger.info(f"Responses type: {type(responses)}")
        # logger.info(f"Responses len: {len(responses)}")
        # logger.info(f"Responses[0] type: {type(responses[0])}")
        # Save response in json file
        for k, response in enumerate(responses):
            try:
                response.save_json(os.path.join(output_path, exp_name, f"{row['filename']}_{k}.json"))
            except Exception as e:
                logger.error(f"Error saving response of {row['filename']}_{k}.json: {e}")
        # logger.info(f"Responses.response: {[response.response for response in responses]}")
        data.at[i, 'recaptions'] = [response.text for response in responses]
        data.at[i, 'responses'] = [response.response for response in responses]
        data.at[i, 'total_used_tokens'] = sum([response.tokens for response in responses])

    data.to_json(os.path.join(output_path, f"memento_{split}_data_{exp_name}.json"), orient='records')

    # Second pass for solving errors
    for i, row in tqdm(data.iterrows(), total=len(data), desc='Generating responses (2nd pass)'):
        responses = chatgpt.retry_get_responses_if_error_in_response(row, base_prompt)
        # Save response in json file
        for k, response in enumerate(responses):
            try:
                response.save_json(os.path.join(output_path, exp_name, f"{row['filename']}_{k}.json"))
            except Exception as e:
                logger.error(f"Error saving response of {row['filename']}_{k}.json: {e}")
        # logger.info(f"Responses.response: {[response.response for response in responses]}")
        data.at[i, 'responses'] = [response.text for response in responses]
        data.at[i, 'recaptions'] = [response.response for response in responses]
        data.at[i, 'total_used_tokens'] = sum([response.tokens for response in responses])

    # data.to_json(os.path.join(output_path, f"memento_{split}_data_{exp_name}.json"))
    # tqdm.pandas(desc='Extracting text')
    # data['answer'] = data['responses'].progress_apply(lambda responses: [response.text for response in responses])
    # # logger.info("Extracting scores...")
    # # train['score_preds'] = train['responses'].progress_apply(lambda responses: get_score_from_responses(responses))
    # logger.info("Computing tokens...")
    # data['total_used_tokens'] = data['responses'].progress_apply(lambda responses: sum([response.tokens for response in responses]))

    logger.info("Saving...")
    data.to_json(os.path.join(output_path, f"memento_{split}_data_{exp_name}.json"), orient='records')

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
