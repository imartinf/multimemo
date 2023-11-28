import json
import os
import logging

import av
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, VivitForVideoClassification, VivitImageProcessor
from tqdm import tqdm

from src.tools.training_utils import VideoDataset
from src.tools.utils import load_videomem


def main(model_folder_path, data_path, video_folder_path, save_path, model_type="vivit"):
    # Print logs to stdout
    logging.basicConfig(level=logging.INFO)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
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

    # Load data
    data = load_videomem(data_path, "video")
    # data = data[:512]
    # data["filename"] = data.index
    # data["filename"] = data["filename"].apply(lambda x: os.path.join(video_folder_path, x))
    data["filename"] = data["video"].apply(lambda x: os.path.join(video_folder_path, x))

    possible_target_columns = ["mem_score", "short-term_memorability", "long-term_memorability"]
    # Match columns in df with possible target columns
    target_columns = [c for c in possible_target_columns if c in data.columns]
    if len(target_columns) == 0:
        data["mem_score"] = 0.5
    else:
        data["mem_score"] = data[target_columns[0]]

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
        dataset,
        batch_size=8,
        shuffle=False,
    )

    # Evaluate model on data
    # The result must be a dict {video_id: prediction}
    model.eval()
    predictions = {}

    for batch, video_ids in tqdm(dataloader):
        video, label = batch
        video = {k: v.to(device) for k, v in video.items() if v is not None}
        with torch.no_grad():
            pred = model(**video).logits.squeeze().cpu().numpy().tolist()
        # Video id is a full path with a filename videoXX.webm, where XX is a number of variable length.
        # Extract the number and convert it to int
        video_ids = [int(os.path.splitext(os.path.basename(id))[0][5:]) for id in video_ids]
        for video_id, p in zip(video_ids, pred):
            predictions[video_id] = p
    # Save predictions as a csv with columns "video_id" and "mem_score"
    predictions = pd.DataFrame.from_dict(predictions, orient="index", columns=["mem_score"])
    predictions.index.name = "video_id"
    # Create save folder if it does not exist
    save_folder = os.path.dirname(save_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    predictions.to_csv(save_path, index=True)


if __name__ == "__main__":
    main(
        model_folder_path="/mnt/rufus_A/models/vivit-finetune-memento-center-segment",
        data_path="/mnt/rufus_A/VIDEOMEM23/testing_set/test-set_videos-captions.txt",
        video_folder_path="/mnt/rufus_A/VIDEOMEM/test-set/Videos/",
        save_path="/mnt/rufus_A/multimemo/runs/vivit-finetune-memento-center-segment/me23mem-THAU-UPM-subtask1-runfViViTCenteredSegment.csv",
    )
