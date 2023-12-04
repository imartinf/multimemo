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
from src.tools.utils import load_videomem


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
    targets = targets[targets.index.isin(predictions.index)]
    # Compute spearman
    spearman = spearmanr(predictions["mem_score"], targets[target_columns[0]].values)
    print(f"Spearman correlation between predictions and targets: {spearman.correlation:.4f}")
    print(f"p-value: {spearman.pvalue:.4f}")
    # Save spearman correlation to a file
    spearman_path = os.path.splitext(save_path)[0] + "_spearman.txt"
    pd.DataFrame({"spearman": [spearman.correlation], "p-value": [spearman.pvalue]}).to_csv(spearman_path, index=False)


@click.command()
@click.option("--model_folder_path", type=str, required=True)
@click.option("--data_path", type=str, required=True)
@click.option("--video_folder_path", type=str, required=True)
@click.option("--save_path", type=str, required=True)
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
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    # Load data
    data = load_videomem(data_path, "video")
    if len(data) > 2000:
        data = data.sample(2000, random_state=42)
    # data = data[:512]
    # data["filename"] = data.index
    # data["filename"] = data["filename"].apply(lambda x: os.path.join(video_folder_path, x))
    # If video is the name of the index, use it as filename
    if "video" not in data.columns and data.index.name == "video":
        data["video"] = data.index
    data["filename"] = data["video"].apply(lambda x: os.path.join(video_folder_path, x))

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
        data["int_id"] = data["video"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))
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
        frame_sample_strategy="center",
        return_video_id=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, pin_memory_device=device
    )

    # Evaluate model on data
    # The result must be a dict {video_id: prediction}
    model.eval()

    for batch, video_ids in tqdm(dataloader):
        predictions = {}
        video, label = batch
        video = {k: v.to(device) for k, v in video.items() if v is not None}
        with torch.no_grad():
            pred = model(**video).logits.squeeze().cpu().numpy().tolist()
        # Video id is a full path with a filename videoXX.webm, where XX is a number of variable length.
        # Extract the number and convert it to int
        video_ids = [int(os.path.splitext(os.path.basename(id))[0][5:]) for id in video_ids]
        # If pred is a float convert it to a list
        if isinstance(pred, float):
            pred = [pred]
        for video_id, p in zip(video_ids, pred):
            predictions[video_id] = p
        # Save predictions as a csv with columns "video_id" and "mem_score" each batch
        predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["mem_score"])
        predictions.index.name = "video_id"
        predictions.to_csv(save_path, index=True, mode="a", header=False)

    if has_target:
        compute_spearman(save_path, data, target_columns)


if __name__ == "__main__":
    main()
