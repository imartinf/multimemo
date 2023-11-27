"""
Extract Saliency features from images and save them to a folder.
"""

import logging
import os

import click
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from src.tools.saliency_processor import SaliencyProcessor
from src.tools.utils import load_memento, load_videomem


def extract_saliency_frame_features(frame_path, saliency_processor, out_path):
    # If the saliency map already exists, load it
    if os.path.exists(out_path):
        return saliency_processor.load_map(out_path).sum()
    saliency_map = saliency_processor.get_saliency_map(frame_path)
    saliency_processor.save(saliency_map, out_path)
    return saliency_map.sum()


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_path_data", type=click.Path(writable=True))
@click.argument("output_path_feat", type=click.Path(writable=True))
@click.argument("feat_cols", nargs=-1, type=click.STRING)
@click.option("--saliency_mode", default="fine_grained", type=click.STRING)
def main(input_filepath, output_path_data, output_path_feat, feat_cols, saliency_mode="fine_grained"):
    """
    Extract Saliency features from images and save them to a folder.
    """
    # logger = logging.getLogger(__name__)
    click.echo(f"Input filepath is: {input_filepath}")
    click.echo(f"Output path for data is: {output_path_data}")
    click.echo(f"Output path for embeddings is: {output_path_feat}")
    feat_cols = list(feat_cols)
    click.echo(f"Feature columns are: {feat_cols}")
    if not os.path.exists(output_path_feat):
        os.makedirs(output_path_feat)
    click.echo(f"Saliency mode is: {saliency_mode}")

    tqdm.pandas()

    dataset = "memento" if "memento" in input_filepath else "videomem"

    if dataset == "videomem":
        data_exp = load_videomem(input_filepath, feat_cols)
    elif dataset == "memento":
        data_exp = load_memento(input_filepath, feat_cols)

    # The feat column must contain path to images
    assert all(os.path.exists(x) for x in data_exp[feat_cols].values.flatten()), "Some images do not exist."
   
    processor = SaliencyProcessor(
        path_to_images=input_filepath,
        saliency_mode=saliency_mode,
    )

    # Append saliency mode to output path
    output_path_feat = os.path.join(output_path_feat, saliency_mode)
    if not os.path.exists(output_path_feat):
        os.makedirs(output_path_feat)

    # Extract features
    for feat_col in feat_cols:
        if feat_col in ["image_path", "frame_path"]:
            if dataset == "memento":
                data_exp[f"{feat_col}_saliency_path"] = data_exp.progress_apply(
                    lambda x: os.path.join(output_path_feat, x["filename"] + "_" + os.path.basename(x[feat_col])),
                    axis=1,
                )
            elif dataset == "videomem":
                data_exp[f"{feat_col}_saliency_path"] = data_exp.progress_apply(
                    lambda x: os.path.join(output_path_feat, os.path.basename(x[feat_col])),
                    axis=1,
                )
            else:
                raise ValueError(f"Unsupported dataset {dataset}.")
            data_exp[f"{feat_col}_saliency"] = data_exp.progress_apply(
                lambda x: extract_saliency_frame_features(x[feat_col], processor, x[f"{feat_col}_saliency_path"]),
                axis=1,
            )
        else:
            raise ValueError(f"Unsupported feature column {feat_col}.")

    # Save data
    data_exp.to_csv(output_path_data, index=False)

    click.echo("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
