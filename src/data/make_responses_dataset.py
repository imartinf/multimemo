# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.tools.chatgpt import ChatGPT


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, file_okay=False))
@click.argument('output_filepath', type=click.Path())
@click.option('-temperature', default=0.9, help='The temperature to use for the response generation.')
def main(input_filepath, output_filepath, temperature):
    """ 
    Script to automatically generate responses using ChatGPT using a dataset of original captions and a prompt.

    :param input_filepath: The path to the input data.
    :type input_filepath: str

    :param output_filepath: The path to the output data.
    :type output_filepath: str

    :param temperature: The temperature to use for the response generation.
    :type temperature: float

    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating responses dataset')

    # Get chatgpt config from environment and set up the ChatGPT object
    gptconfig = {
        'MODEL': os.environ.get('MODEL'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
        'OPENAI_API_VERSION': os.environ.get('OPENAI_API_VERSION')
    }

    gptconfig['TEMPERATURE'] = temperature

    chatgpt = ChatGPT(gptconfig)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
