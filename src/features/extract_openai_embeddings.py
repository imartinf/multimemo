"""
Extract OpenAI Embeddings and save them to a folder.
"""

import logging
import click
import os
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.tools.chatgpt import ChatGPT

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_path_data', type=click.Path(writable=True))
@click.argument('output_path_emb', type=click.Path(writable=True))
@click.argument('text_cols', nargs=-1, type=click.STRING)

def main(input_filepath, output_path_data, output_path_emb, text_cols):
    """
    Extract OpenAI Embeddings and save them to a folder.
    """
    logger = logging.getLogger(__name__)
    click.echo(f"Input filepath is: {input_filepath}")
    click.echo(f"Output path for data is: {output_path_data}")
    click.echo(f"Output path for embeddings is: {output_path_emb}")
    text_cols = list(text_cols)
    click.echo(f"Text columns are: {text_cols}")
    if not os.path.exists(output_path_data):
        os.makedirs(output_path_data)
    if not os.path.exists(output_path_emb):
        os.makedirs(output_path_emb)


    tqdm.pandas()

    # Load data
    data = pd.read_json(input_filepath)
    data_exp = data.explode(text_cols, ignore_index=True)
    for text_col in text_cols:
        data[f"{text_col}_emb_path"] = ''
    click.echo(f"Loaded {len(data_exp)} samples.")

    # Get chatgpt config from environment and set up the ChatGPT object
    gptconfig = {
        'MODEL': os.environ.get('MODEL'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
        'OPENAI_API_VERSION': os.environ.get('OPENAI_API_VERSION'),
        'ENGINE': os.environ.get('ENGINE')
    }

    chatgpt = ChatGPT(gptconfig, logger)

    # Extract embeddings
    last_filename = ''
    i = 0
    for text_col in text_cols:
        click.echo(f"Extracting embeddings for {text_col}.")
        for ind, row in tqdm(data_exp.iterrows(), total=len(data_exp)):
            if row['filename'] == last_filename:
                i += 1
            else:
                i = 0
                last_filename = row['filename']
            try:
                base, _ = os.path.splitext(last_filename)
            except Exception as e:
                print(e)
                print(base)
            save_path = os.path.join(output_path_emb, f"{text_col}-{base}-{i}.npy")
            # Find if file already exists
            if os.path.exists(save_path) and np.load(save_path, allow_pickle=True).dtype == np.float64:
                data_exp.at[ind, f'{text_col}_emb_path'] = str(save_path)
                continue
            emb = chatgpt.get_embedding(row[text_col])['data'][0]['embedding']
            # Save embedding
            data_exp.at[ind, f'{text_col}_emb_path'] = str(save_path)
            np.save(save_path, emb)

    # Save data
    data_exp.to_json(os.path.join(output_path_data, 'memento_data_recaption_exp_emb.json'))

    click.echo("Done!")

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    
    main()