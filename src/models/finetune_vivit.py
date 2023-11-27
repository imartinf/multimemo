import logging
import os
import random
import time

import click
import evaluate
import pandas as pd
import torch
import torch.nn as nn
import transformers
import psutil
from tqdm import tqdm
from transformers import TrainingArguments, VivitConfig, VivitForVideoClassification, VivitImageProcessor

import wandb

from src.tools.training_utils import CustomTrainer, VideoDataset


# Function to generate a random number between 0 and 1
def generate_random_number():
    return random.uniform(0, 1)


# Function to scan a folder and create a dictionary
def create_dictionary_from_folder(folder_path):
    video_dict = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mkv", ".mov")):
                video_path = os.path.join(root, file)
                random_number = generate_random_number()
                video_dict[video_path] = random_number
    return video_dict


def collate_fn(examples):
    pixel_values = torch.stack([example[0]["pixel_values"] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_spearman(eval_pred):
    metric = evaluate.load("spearmanr")
    try:
        logits, labels = eval_pred
    except BaseException:
        try:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        except BaseException:
            logging.error("Error getting logits and labels")
            exit()
    return metric.compute(predictions=logits, references=labels)


def model_init(trial):
    config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
    config.num_labels = 1
    config.num_frames = 16
    if trial is None or "num_frames" not in trial.keys():
        config.num_frames = 16
        config.video_size = [16, 224, 224]
    if trial is not None:
        for k, v in trial.items():
            # Check if keys are in config
            if k in config.to_dict():
                setattr(config, k, v)
        if config.num_frames != config.video_size[0]:
            config.video_size[0] = config.num_frames
    # Replace config values for trial values
    model = VivitForVideoClassification(config)
    return model


def model_init_finetune(trial):
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400", num_labels=1, ignore_mismatched_sizes=True
    )
    # Freeze all layers except the last one
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    return model


def wandb_hp_space(trial):
    # Get a unique trial name across all trials
    trial_name = wandb.util.generate_id()
    return {
        "method": "grid",
        "metric": {"name": "eval/spearmanr", "goal": "maximize"},
        "parameters": {
            # Epochs are ints
            "num_train_epochs": {"value": 10},
            # "per_device_train_batch_size": {"value": 4},
            "warmup_ratio": {"value": 0.4},
            # Set evaluation batch size equal to training batch size
            # "per_device_eval_batch_size": {"ref": "per_device_train_batch_size"},
            "gradient_accumulation_steps": {"values": [4, 8, 16]},
            # "warmup_ratio": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            "learning_rate": {"value": 1e-04},
            "weight_decay": {"value": 0.0001},
            # "num_frames": {"values": [2, 4, 8, 16, 32]},
        },
        "name": f"validation-gradient-accumulation-steps-{trial_name}-gth11",
    }


def check_cpu_usage(logger):
    # Get current CPU usage
    cpu_usage = psutil.cpu_percent()
    logger.info(f"Current CPU usage: {cpu_usage}%")
    # If CPU usage is greater than 90%, sleep for 10 seconds
    while cpu_usage > 90:
        logger.info("CPU usage is greater than 90%. Sleeping for 3 seconds.")
        time.sleep(3)
        cpu_usage = psutil.cpu_percent()
        logger.info(f"Current CPU usage: {cpu_usage}%")


@click.command()
@click.argument("base_dir", type=click.Path(exists=True))
@click.argument("exp_name", type=click.STRING)
@click.argument("log_dir", type=click.Path(writable=True))
# video_dir defaults to base_dir
@click.option("--video_dir", type=click.Path(exists=True), default=None)
@click.option("--method", type=click.STRING, default="pytorch")
@click.option("--param_search", type=click.BOOL, default=False)
@click.option("--finetune", type=click.BOOL, default=False)
@click.option("--batch_size", type=click.INT, default=4)
@click.option("--learning_rate", type=click.FLOAT, default=5e-5)
@click.option("--num_epochs", type=click.INT, default=1)
@click.option("--sample", type=click.FLOAT, default=1)
def main(
    base_dir,
    exp_name,
    log_dir,
    video_dir,
    method,
    param_search,
    finetune,
    learning_rate,
    batch_size,
    num_epochs,
    sample,
):
    # Setup logging using click
    log_file = os.path.join(log_dir, f"logger-{exp_name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("running experiment")
    logger.info(f"base dir is: {base_dir}")
    logger.info(f"exp name is: {exp_name}")
    logger.info(f"log dir is: {log_dir}")
    logger.info(f"video dir is: {video_dir}")
    logger.info(f"method is: {method}")
    logger.info(f"finetune is: {finetune}")
    logger.info(f"batch size is: {batch_size}")
    logger.info(f"num epochs is: {num_epochs}")
    logger.info(f"sample is: {sample}")
    logger.info(f"param search is: {param_search}")

    logger.info(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info("Resources: ")
        logger.info(torch.cuda.memory_summary())
    try:
        logger.info(f"Available GPU RAM: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")
    except RuntimeError:
        logger.info("No GPU available")
    logger.info(f"CPU RAM: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"Python version: {os.popen('python --version').read()}")

    BASE_DIR = base_dir
    if not video_dir:
        video_dir = base_dir
    # av.logging.set_level(av.logging.ERROR)

    # check_cpu_usage(logger)
    model_ckpt = "google/vivit-b-16x2-kinetics400"
    image_processor = VivitImageProcessor.from_pretrained(model_ckpt)
    # check_cpu_usage(logger)
    if torch.cuda.is_available():
        logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(torch.device('cuda'))/1024**3} GB")

    config = VivitConfig.from_pretrained(model_ckpt)
    config.num_labels = 1
    config.num_frames = 16

    if not param_search:
        if not finetune:
            model_ft = VivitForVideoClassification(config)
        else:
            model_ft = VivitForVideoClassification.from_pretrained(
                model_ckpt, num_labels=1, ignore_mismatched_sizes=True
            )
            # check_cpu_usage(logger)
            if torch.cuda.is_available():
                logger.info(
                    f"Current GPU memory usage: {torch.cuda.memory_allocated(torch.device('cuda'))/1024**3} GB"
                )
            # Freeze all layers except the last one
            # for param in model_ft.parameters():
            #     param.requires_grad = False
            # for param in model_ft.classifier.parameters():
            #     param.requires_grad = True
        num_frames_to_sample = model_ft.config.num_frames
        sample_rate = image_processor.resample
        model_ft.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    else:
        num_frames_to_sample = config.num_frames
        sample_rate = image_processor.resample
        model_ft = None

    video_path = os.path.join(video_dir, "videos/")
    train_data = pd.read_json(os.path.join(video_dir, "memento_train_data.json")).sample(frac=sample)
    test_data = pd.read_json(os.path.join(video_dir, "memento_val_data.json")).sample(frac=sample)
    # check_cpu_usage(logger)

    train_data["filename"] = train_data["filename"].apply(lambda x: video_path + x)
    test_data["filename"] = test_data["filename"].apply(lambda x: video_path + x)

    # Search inside the video folder for videos ending in _resampled.mp4 and substitute the original video for it
    # This is done to avoid errors with some videos
    num_of_resampled_videos = 0
    for filename in pd.concat([train_data, test_data], ignore_index=True).filename.values:
        # check_cpu_usage(logger)
        if filename.endswith("_resampled.mp4"):
            original_filename = filename.replace("_resampled.mp4", ".mp4")
            if os.path.exists(original_filename):
                num_of_resampled_videos += 1
                logging.info(f"Replacing {filename} with {original_filename}")
                train_data["filename"] = train_data["filename"].replace(filename, original_filename)
                test_data["filename"] = test_data["filename"].replace(filename, original_filename)
    logging.info(f"Replaced {num_of_resampled_videos} videos with their resampled versions.")

    # check_cpu_usage(logger)

    train_dataset = VideoDataset(
        train_data, "filename", "mem_score", num_frames_to_sample, sample_rate, video=True, processor=image_processor
    )
    test_dataset = VideoDataset(
        test_data, "filename", "mem_score", num_frames_to_sample, sample_rate, video=True, processor=image_processor
    )

    # check_cpu_usage(logger)

    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-memento-{exp_name}"

    # Train using native pytorch
    if method == "pytorch":
        logger.info("Training using native pytorch")
        metric = evaluate.load("spearmanr")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check_cpu_usage(logger)
        # Log GPU usage
        if torch.cuda.is_available():
            logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(device)/1024**3} GB")
        model_ft.to(device)
        if torch.cuda.is_available():
            logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(device)/1024**3} GB")
        # check_cpu_usage(logger)

        optimizer = torch.optim.AdamW(model_ft.parameters(), lr=5e-5)
        criterion = nn.MSELoss()

        train_loader = train_dataset.load("train", batch_size=batch_size)
        test_loader = test_dataset.load("test", batch_size=batch_size)

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch}")
            model_ft.train()
            for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model_ft(**inputs)
                    loss = criterion(outputs.logits.squeeze(), labels.squeeze())
                    loss.backward()
                    optimizer.step()
                    if i % 10 == 0:
                        logging.info(f"Epoch {epoch} | Batch {i} | Loss: {loss.item()}")
            model_ft.eval()
            with torch.set_grad_enabled(False):
                for i, (inputs, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    outputs = model_ft(**inputs)
                    preds = outputs.logits.squeeze()
                    logging.info(f"Epoch {epoch} | Batch {i} | Predictions: {preds}")
                    logging.info(f"Epoch {epoch} | Batch {i} | Labels: {labels.squeeze()}")
                    spearman = metric.compute(predictions=preds, references=labels.squeeze())
                    logging.info(f"Epoch {epoch} | Spearman: {spearman}")

    elif method == "transformers":
        # Train using transformers Trainer
        args = TrainingArguments(
            new_model_name,
            remove_unused_columns=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            auto_find_batch_size=True,
            # per_device_train_batch_size=batch_size,
            # per_device_eval_batch_size=batch_size,
            # gradient_accumulation_steps=16,
            # tf32=True,
            # warmup_ratio=0.1,
            logging_steps=10,
            logging_dir=os.path.join(BASE_DIR, "logs", new_model_name),
            load_best_model_at_end=True,
            metric_for_best_model="spearmanr",
            report_to="wandb",
            run_name=new_model_name,
        )

        trainer = CustomTrainer(
            None,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=image_processor,
            data_collator=collate_fn,
            compute_metrics=compute_spearman,
            model_init=model_init if not finetune else model_init_finetune,
        )

        if not param_search:
            trainer.train()
        else:
            # Hyperparameter search
            best_trial = trainer.hyperparameter_search(
                direction="maximize",
                backend="wandb",
                hp_space=wandb_hp_space,
                n_trials=30,
                # compute_objective=compute_spearman,
                # hp_name=new_model_name,
                project="training-video-transformers-v3",
                metric="eval/spearmanr",
            )
            # Save results
            logging.info(best_trial)

    logging.info("Finished training")


if __name__ == "__main__":
    main()
