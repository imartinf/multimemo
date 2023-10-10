import json
import os

import click
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm


def save_json(json_data, path2save, json_indent):
    messages_JSON = json.dumps(json_data, indent=json_indent)
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path2save), exist_ok=True)
    with open(path2save, "w") as write_file:
        write_file.write(messages_JSON)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        click.echo(f"Directory {dir_path} created.")
    else:
        click.echo(f"Directory {dir_path} already exists.")

def load_data(path):
    """
    Loads data from path. Path can direct to a file or a directory.
    If path is a file, it loads and returns the file.
    If path is a directory, it searches for split keywords in the name and returns
    a dictionary with the splits as keys and the files as values.

    The data is returned as DataFrames

    Automatically infers data extension and loads accordingly.

    :param path: The path to the data.
    :type path: str

    :return: The data.
    :rtype: dict
    """

    # Load data
    if os.path.isfile(path):
        # Load file
        data = load_file(path)
    else:
        # Load directory
        data = load_dir(path)

    return data

def load_file(path):
    """
    Loads a file from path.

    Automatically infers data extension and loads accordingly.

    :param path: The path to the file.
    :type path: str

    :return: The data.
    :rtype: DataFrame
    """
    # Load data
    if path.endswith('.csv'):
        data = pd.read_csv(path)
    elif path.endswith('.json'):
        data = pd.read_json(path, orient='records')
    elif path.endswith('.pkl'):
        data = pd.read_pickle(path)
    else:
        raise ValueError(f"File extension not supported: {path}")

    return data

def load_dir(path):
    """
    Loads a directory from path.

    Automatically infers data extension and loads accordingly.

    :param path: The path to the directory.
    :type path: str

    :return: The data.
    :rtype: dict
    """
    # Raise exception if path is not a directory
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")
    else:
        # Get all files in directory
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # Get splits
        splits = set([f.split('_')[1] for f in files])
        # Load files
        data = {}
        for split in splits:
            # Get path which contains split
            split_path = [f for f in files if split in f]
            # Load file
            data[split] = load_file(os.path.join(path, split_path[0]))
    return data

def load_config(path):
    """
    Loads a config file from path.

    Automatically infers data extension and loads accordingly.

    :param path: The path to the config file.
    :type path: str

    :return: The config.
    :rtype: dict
    """

    # Load config
    if path.endswith('.json'):
        with open(path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Config file extension not supported: {path}")

    return config

def explode_data(data, columns):
    """
    Explodes the dataframes that are passed as values of the data dictionary on the columns passed as a list.

    :param data: The data.
    :type data: dict

    :param columns: The columns to explode.
    :type columns: list

    :return: The data.
    :rtype: dict
    """

    # Explode each dataframe on captions and recaptions simultaneously
    for split, df in data.items():
        # Check if columns are in dataframe
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column {column} not in dataframe")
        # Explode dataframe
        data[split] = df.explode(columns, ignore_index=True)
    return data

def build_memarray_from_files(files):
    """
    Build a memory array from a list of files.
    """
    try:
        arr = open_memmap(files[0])
    except Exception as e:
        click.echo(f"Could not load file {files[0]}")
        raise e
    for file in tqdm(files[1:]):
        arr = np.vstack((arr, open_memmap(file)))
    return arr

def z_score(x, mean, std):
    """
    Z-score a value.
    """
    return (x - mean) / std