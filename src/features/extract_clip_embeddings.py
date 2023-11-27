"""
Extract CLIP Embeddings and save them to a folder.
"""

from ast import literal_eval
import logging
import click
import os
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
import torch
from tqdm import tqdm

from src.models.clip import CLIP, get_textual_embeddings, get_visual_embeddings
from src.tools.utils import load_memento, load_videomem


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_path_data", type=click.Path(writable=True))
@click.argument("output_path_emb", type=click.Path(writable=True))
@click.argument("feat_cols", nargs=-1, type=click.STRING)
@click.option("--checkpoint", type=click.STRING, default=None)
def main(input_filepath, output_path_data, output_path_emb, feat_cols, checkpoint):
    """
    Extract CLIP Embeddings and save them to a folder.
    """
    # logger = logging.getLogger(__name__)
    click.echo(f"Input filepath is: {input_filepath}")
    click.echo(f"Output path for data is: {output_path_data}")
    click.echo(f"Output path for embeddings is: {output_path_emb}")
    feat_cols = list(feat_cols)
    click.echo(f"Feature columns are: {feat_cols}")
    if not os.path.exists(output_path_emb):
        os.makedirs(output_path_emb)
    if checkpoint is not None:
        click.echo(f"Checkpoint is: {checkpoint}")

    tqdm.pandas()

    dataset = "memento" if "memento" in input_filepath else "videomem"

    if dataset == "videomem":
        data_exp = load_videomem(input_filepath, feat_cols)
    elif dataset == "memento":
        raise NotImplementedError("Memento dataset is not implemented yet.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model definition
    if checkpoint is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    else:
        model = CLIP(output_embed=True)
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Extract embeddings
    if not isinstance(feat_cols, list):
        feat_cols = [feat_cols]
    for feat_col in feat_cols:
        if feat_col in ["image_path", "frame_path"]:
            # Remove duplicates if they belong to the same filename
            data_exp_ = data_exp.drop_duplicates(subset=["filename", feat_col]).reset_index(drop=True)
            embs = get_visual_embeddings(
                model,
                processor,
                data_exp_[feat_col].tolist(),
                device,
            )
        elif feat_col in ["captions", "caption"]:
            # Remove duplicates if they belong to the same filename
            data_exp_ = data_exp.drop_duplicates(subset=["filename", feat_col]).reset_index(drop=True)
            embs = get_textual_embeddings(
                model,
                processor,
                data_exp_[feat_col].tolist(),
                device,
            )
        else:
            raise ValueError(f"Unknown feature column {feat_col}.")
        # Embeddings are numpy arrays, save each of them to a file in the output folder
        if not os.path.exists(os.path.join(output_path_emb, feat_col)):
            os.makedirs(os.path.join(output_path_emb, feat_col))
        # There is an embedding for every unique caption, but the captions are repeated for every frame
        # Assign the embedding to every frame of the same video
        for ind, row in data_exp_.iterrows():
            if row[feat_col] != "":
                base = (
                    os.path.basename(row["filename"])
                    if feat_col in ["captions", "caption"]
                    else os.path.basename(row[feat_col])
                )
                # Remove extension in save_path with .npy
                save_path = os.path.join(output_path_emb, feat_col, f"{feat_col}-{base}")
                save_path = os.path.splitext(save_path)[0] + ".npy"
                with open(save_path, "wb") as f:
                    np.save(f, embs[ind])
                data_exp_.loc[ind, f"{feat_col}_emb_path"] = save_path
        # Merge data_exp_ with data_exp on filename
        data_exp = pd.merge(data_exp, data_exp_[["filename", feat_col, f"{feat_col}_emb_path"]], on=["filename", feat_col], how="left")
    # Save data
    data_exp.to_csv(output_path_data, index=False)

    click.echo("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
