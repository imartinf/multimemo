import json
import logging
import os
import sys

import click
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, VivitForVideoClassification, VivitImageProcessor

from src.tools.training_utils import VideoDataset
from src.tools.utils import load_videomem, load_memento
from src.tools.video_processing import create_segment_database
from src.models.vivit import CustomVivit


def evaluate_all_video_segments(model, batch, video_ids, device):
    video = [segment[0] for segment in batch[0]]
    video = [{k: v.to(device) for k, v in segment.items() if v is not None} for segment in video]
    with torch.no_grad():
        return model(video).logits.squeeze().cpu().numpy().tolist()


def evaluate_single_segment(model, batch, video_ids, device):
    video, label = batch
    video = {k: v.to(device) for k, v in video.items() if v is not None}
    with torch.no_grad():
        return model(**video).logits.squeeze().cpu().numpy().tolist()


def compute_spearman(save_path, data, target_columns):
    print("Computing spearman correlation")
    print(f"Using {target_columns[0]} as target")
    print(f"Loading predictions from {save_path}")
    print(f"Loading targets from {data}")
    print(f"Number of videos: {len(data)}")
    print(f"Number of videos with predictions: {len(pd.read_csv(save_path, index_col='video_id'))}")
    # Compute spearman between predictions and targets
    predictions = pd.read_csv(save_path, index_col="video_id")
    targets = data[target_columns]
    # Remove videos that were not processed
    if len(predictions) != len(targets):
        print(f"Removing {len(targets) - len(predictions)} videos that were not processed")
        targets = targets[targets.index.isin(predictions.index)]
    # Compute spearman
    spearman = spearmanr(predictions["mem_score"], targets[target_columns[0]].values)
    print(f"Spearman correlation between predictions and targets: {spearman.correlation:.4f}")
    print(f"p-value: {spearman.pvalue:.4f}")
    # Save spearman correlation to a file
    spearman_path = os.path.splitext(save_path)[0] + "_spearman.txt"
    pd.DataFrame({"spearman": [spearman.correlation], "p-value": [spearman.pvalue]}).to_csv(spearman_path, index=False)


@click.command()
@click.argument("model_folder_path", type=str)
@click.argument("data_path", type=str)
@click.argument("video_folder_path", type=str)
@click.argument("save_path", type=str)
@click.option("--model_type", type=str, default="vivit")
def main(model_folder_path, data_path, video_folder_path, save_path, model_type="vivit"):
    # Print logs to stdout
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")
    # device = "cpu"
    model_config_path = os.path.join(model_folder_path, "config.json")
    model_processor_config_path = os.path.join(model_folder_path, "preprocessor_config.json")
    model_weights_path = os.path.join(model_folder_path, "pytorch_model.bin")

    # Load model config
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    # Load model preprocessor config
    with open(model_processor_config_path, "r") as f:
        model_processor_config = json.load(f)

    if model_type == "vivit":
        processor = VivitImageProcessor(**model_processor_config)
        model = VivitForVideoClassification(AutoConfig.from_pretrained(model_config_path)).to(device)
        model.classifier = nn.Sequential(model.classifier, nn.Sigmoid())
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    elif model_type == "custom_vivit":
        processor = VivitImageProcessor(**model_processor_config)
        model = CustomVivit(AutoConfig.from_pretrained(model_config_path)).to(device)
        model.vivit.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    # Load data
    data = load_videomem(data_path, "video") if "videomem" in data_path else load_memento(data_path, "filename")
    if "videomem" in data_path and len(data) > 2000:
        data = data.sample(2000, random_state=42)
    else:
        data = data[:1500].reset_index(drop=True)
    # data = data[:32]
    # data["filename"] = data.index
    # data["filename"] = data["filename"].apply(lambda x: os.path.join(video_folder_path, x))
    # If video is the name of the index, use it as filename
    if "video" not in data.columns and data.index.name == "video":
        data["video"] = data.index
    data["filename"] = (
        data["video"].apply(lambda x: os.path.join(video_folder_path, x))
        if "video" in data.columns
        else data["filename"].apply(lambda x: os.path.join(video_folder_path, x))
    )

    # data = create_segment_database(data, "filename", model_config["num_frames"], 15)
    possible_target_columns = ["mem_score", "short-term_memorability"]
    # Match columns in df with possible target columns
    target_columns = [c for c in possible_target_columns if c in data.columns]
    has_target = False
    if len(target_columns) == 0:
        data["mem_score"] = 0.5
    else:
        assert len(target_columns) == 1, "Multiple target columns found"
        data["mem_score"] = data[target_columns[0]]
        print(f"Using {target_columns[0]} as target")
        has_target = True

    # Check if output file already exists
    if os.path.exists(save_path):
        # Check which videos have already been processed and remove them from the data
        try:
            processed_videos = pd.read_csv(save_path, index_col="video_id")
        except ValueError:
            processed_videos = pd.read_csv(save_path, index_col=0)
            processed_videos.index.name = "video_id"
            # Assign the column name as mem_score
            processed_videos.columns = ["mem_score"]
            # Save
            processed_videos.to_csv(save_path, index=True)
        initial_data_len = len(data)
        data["int_id"] = (
            data["video"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))
            if "video" in data.columns
            else data["filename"].apply(
                lambda x: os.path.basename(x).split(".")[0]
            )  # Just get the filename and remove extension
        )
        data = data[~data["int_id"].isin(processed_videos.index)]
        logging.info(f"Skipping {len(processed_videos)} videos that have already been processed")
        assert len(data) + len(processed_videos) == initial_data_len, "Data length mismatch"
    else:
        # Create save folder if it does not exist
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    dataset = VideoDataset(
        data,
        "filename",
        "mem_score",
        model_config["num_frames"],
        processor.resample,
        video=True,
        processor=processor,
        frame_sample_strategy="all-segments",
        return_video_id=True,
        eval=True,
    )

    # Evaluate model on data
    # The result must be a dict {video_id: prediction}
    model.eval()

    for batch in tqdm(dataset, total=len(dataset)):
        # Save predictions as a csv with columns "video_id" and "mem_score" each batch
        predictions = {}
        if dataset.eval:
            video_ids = batch[1]
            pred = evaluate_all_video_segments(model, batch, video_ids, device)
        else:
            batch, video_ids = batch
            pred = evaluate_single_segment(model, batch, video_ids, device)
        # Video id is a full path with a filename videoXX.webm, where XX is a number of variable length.
        # Extract the number and convert it to int
        if "videomem" in data_path:
            video_ids = [int(os.path.splitext(os.path.basename(id))[0][5:]) for id in video_ids]
        # If pred is a float convert it to a list
        if isinstance(pred, float):
            pred = [pred]
        if isinstance(video_ids, int) or isinstance(video_ids, str):
            video_ids = [video_ids]
        for video_id, p in zip(video_ids, pred):
            predictions[video_id] = p
        predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["mem_score"])
        predictions.index.name = "video_id"
        # Save predictions
        if os.path.exists(save_path):
            predictions.to_csv(save_path, mode="a", header=False)
        else:
            predictions.to_csv(save_path, index=True)

    if has_target:
        compute_spearman(save_path, data, target_columns)


if __name__ == "__main__":
    main()
