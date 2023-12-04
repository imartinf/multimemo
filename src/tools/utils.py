import json
import os

from ast import literal_eval
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
    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".json"):
        data = pd.read_json(path, orient="records")
    elif path.endswith(".pkl"):
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
        splits = set([f.split("_")[1] for f in files])
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
    if path.endswith(".json"):
        with open(path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Config file extension not supported: {path}")

    return config


def load_videomem(input_filepath, feat_cols):
    # Load data
    data_ext = os.path.splitext(input_filepath)[1].replace(".", "")
    data_reader = {"csv": pd.read_csv, "json": pd.read_json, "txt": pd.read_csv}
    add_args = {
        "csv": {"converters": {"frame_path": literal_eval}, "index_col": 0},
        "json": {"orient": "records"},
        "txt": {"sep": "\t", "names": ["video", "caption"]},
    }
    data = data_reader[data_ext](input_filepath, **add_args[data_ext])
    if "caption" in data.columns and "caption" in feat_cols:
        data["caption"] = data["caption"].apply(lambda x: x.replace("-", " "))
    # Subsample for debugging
    # data = data.sample(256).reset_index(drop=True)
    if "frame_path" in feat_cols:
        data = data.explode("frame_path", ignore_index=True).reset_index(drop=True)
        data["frame_path"] = data["frame_path"].apply(
            lambda x: os.path.join("/mnt/rufus_A/VIDEOMEM/dev-set/all_frames", x)
        )
        data = subsample_3frames(data)
    # if not isinstance(feat_cols, list):
    #     feat_cols = [feat_cols]
    # for feat_col in feat_cols:
    #     data[f"{feat_col}_emb_path"] = ""
    click.echo(f"Loaded {len(data)} samples.")
    return data


def load_memento(input_filepath, feat_cols):
    """
    WIP
    """
    try:
        if "train" in input_filepath:
            # Load both train and val and concat them
            data1 = load_data(input_filepath)
            data2 = (
                load_data(input_filepath.replace("train", "val"))
                if os.path.exists(input_filepath.replace("train", "val"))
                else pd.DataFrame()
            )
            data = pd.concat([data1, data2]).reset_index(drop=True)
        elif "val" in input_filepath:
            # Load both train and val and concat them
            data2 = load_data(input_filepath)
            data1 = load_data(input_filepath.replace("val", "train"))
            data = pd.concat([data1, data2]).reset_index(drop=True)
        else:
            data = load_data(input_filepath)
        # Apply literal_eval to feat_cols
        if isinstance(feat_cols, list):
            for f in feat_cols:
                try:
                    data[f] = data[f].apply(literal_eval)
                except ValueError:
                    pass
        elif isinstance(feat_cols, str):
            try:
                data[feat_cols] = data[feat_cols].apply(literal_eval)
            except ValueError:
                pass
        else:
            raise ValueError(f"feat_cols must be a list or a string, got {type(feat_cols)}")
        # Subsample if there are more than 3 unique frame_paths per video
        # if data.groupby("filename")["frame_path"].nunique().max() > 3:
        #     data = subsample_3frames(data)
        # Explode data if it contains lists
        first_feat_col = feat_cols[0] if isinstance(feat_cols, list) else feat_cols
        if isinstance(data[first_feat_col].iloc[0], list):
            data = data.explode(first_feat_col).reset_index(drop=True)
            click.echo("Exploded data.")
        # Remove duplicates if data is already exploded
        else:
            if len(data) > data[first_feat_col].nunique():
                data = data.drop_duplicates(subset=first_feat_col).reset_index(drop=True)
                click.echo("Removed duplicates.")
        if "frame_path" in data.columns:
            data["frame_path"] = data["frame_path"].apply(lambda x: os.path.join("/mnt/rufus_A/Memento10k/videos", x))
        # Subsample for debugging
        # data_exp = data_exp.sample(256).reset_index(drop=True)
        # if isinstance(feat_cols, list):
        #     for feat_col in feat_cols:
        #         data[f"{feat_col}_emb_path"] = ""
        # else:
        #     data[f"{feat_cols}_emb_path"] = ""
        click.echo(f"Loaded {len(data)} samples.")
        return data
    except ValueError as e:
        click.echo(f"Error loading data: {e}")
        raise e


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


def subsample_3frames(data: pd.DataFrame) -> pd.DataFrame:
    """
    Get the first, middle and last frames from each video by filtering the original dataset
    TODO: (imartinf) generalize to n fps.
    """
    # Group by filename and order each group by framepath
    data = (
        data.groupby(["filename"])
        .apply(lambda x: x.sort_values(["frame_path"], ascending=True))
        .reset_index(drop=True)
    )

    return data.groupby(["filename"]).apply(lambda x: x.iloc[[0, int(len(x) / 2), -1]]).reset_index(drop=True)
