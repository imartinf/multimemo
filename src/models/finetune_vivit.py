import logging
import os
import random
import time

import av
import click
import evaluate
import numpy as np
import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image
import psutil
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, EarlyStoppingCallback, Trainer, TrainingArguments,
                          VivitConfig, VivitForVideoClassification,
                          VivitImageProcessor, VivitModel)

import wandb

from tools.training_utils import VideoDataset, CustomTrainer
from tools.video_processing import create_segment_database

# Function to generate a random number between 0 and 1
def generate_random_number():
    return random.uniform(0, 1)

# Function to scan a folder and create a dictionary
def create_dictionary_from_folder(folder_path):
    video_dict = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_path = os.path.join(root, file)
                random_number = generate_random_number()
                video_dict[video_path] = random_number
    return video_dict

def collate_fn(examples):

    pixel_values = torch.stack(
        [example[0]["pixel_values"] for example in examples]
    )
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_spearman(eval_pred):
    metric = evaluate.load("spearmanr")
    try:
        logits, labels = eval_pred
    except:
        try:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        except:
            logging.error("Error getting logits and labels")
            exit()
    return metric.compute(predictions=logits, references=labels)

def model_init(trial):
    config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
    config.num_labels = 1
    config.num_frames = 15
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    if trial is None or 'num_frames' not in trial.keys():
        config.num_frames = 15
        config.video_size = [15, 224, 224]
    if trial is not None:
        for k, v in trial.items():
        # Check if keys are in config
            if k in config.to_dict():
                setattr(config, k, v)
        if config.num_frames != config.video_size[0]:
            config.video_size[0] = config.num_frames
    # Replace config values for trial values
    wandb.config.update(config)
    model = VivitForVideoClassification(config)
    return model

def model_init_finetune(trial):
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400", num_labels=1, ignore_mismatched_sizes=True)
    # # Freeze all layers except last encoder and regression head
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    # # Unfreeze last three transformer blocks
    # for param in model.vivit.encoder.layer[-5:].parameters():
    #     param.requires_grad = True
    # Append sigmoid activation to regression head
    model.classifier = nn.Sequential(model.classifier, nn.Sigmoid())
    return model

def wandb_hp_space(trial):
    # Get a unique trial name across all trials
    trial_name = wandb.util.generate_id()
    return {
        "method": "grid",
        "metric": {
            "name": "eval/spearmanr",
            "goal": "maximize"
        },
        "parameters": {
            # Epochs are ints
            # "num_train_epochs": {"value": 10},
            # "per_device_train_batch_size": {"value": 4},
            # "warmup_ratio": {'value': 0.4},
            # Set evaluation batch size equal to training batch size
            # "per_device_eval_batch_size": {"ref": "per_device_train_batch_size"},
            # "gradient_accumulation_steps": {"values": [1, 2, 4, 8, 16]},
            # "warmup_ratio": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            "learning_rate": {"value": 1e-04},
            # "weight_decay": {"value": 0.0001},
            # "num_frames": {"values": [2, 4, 8, 16, 32]},
        },
        "name": f"validation-gradient-accumulation-steps-finetune-{trial_name}-magerit",
    }

def check_cpu_usage(logging):
    # Get current CPU usage
    cpu_usage = psutil.cpu_percent()
    logging.info(f"Current CPU usage: {cpu_usage}%")
    # If CPU usage is greater than 90%, sleep for 10 seconds
    while cpu_usage > 90:
        logging.info("CPU usage is greater than 90%. Sleeping for 3 seconds.")
        time.sleep(3)
        cpu_usage = psutil.cpu_percent()
        logging.info(f"Current CPU usage: {cpu_usage}%")

@click.command()
@click.argument('base_dir', type=click.Path(exists=True))
@click.argument('exp_name', type=click.STRING)
@click.argument('log_dir', type=click.Path(writable=True))
# video_dir defaults to base_dir
@click.option('--video_dir', type=click.Path(exists=True), default=None)
@click.option('--method', type=click.STRING, default='pytorch')
@click.option('--param_search', type=click.BOOL, default=False)
@click.option('--finetune', type=click.BOOL, default=False)
@click.option('--batch_size', type=click.INT, default=4)
@click.option('--learning_rate', type=click.FLOAT, default=1e-5)
@click.option('--num_epochs', type=click.INT, default=50)
@click.option('--sample', type=click.FLOAT, default=1)
@click.option('--frame_sample_strategy', type=click.STRING, default="uniform")
@click.option('--saliency_scores', type=click.Path(exists=True), default=None)
def main(base_dir, exp_name, log_dir, video_dir, method, param_search, finetune, learning_rate, batch_size, num_epochs, sample, frame_sample_strategy, saliency_scores):
    # Setup logging using click
    log_file = os.path.join(log_dir, f"logging-{exp_name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # transformers.logging.set_verbosity_debug()
    logging.info('running experiment')
    logging.info(f'base dir is: {base_dir}')
    logging.info(f'exp name is: {exp_name}')
    logging.info(f'log dir is: {log_dir}')
    logging.info(f'video dir is: {video_dir}')
    logging.info(f'method is: {method}')
    logging.info(f'finetune is: {finetune}')
    logging.info(f'batch size is: {batch_size}')
    logging.info(f'num epochs is: {num_epochs}')
    logging.info(f'sample is: {sample}')
    logging.info(f'param search is: {param_search}')
    logging.info(f'frame sample strategy is: {frame_sample_strategy}')
    logging.info(f'saliency scores is: {saliency_scores}')

    logging.info(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info("Resources: ")
        logging.info(torch.cuda.memory_summary())
    try:
        logging.info(f"Available GPU RAM: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")
    except RuntimeError:
        logging.info("No GPU available")
    logging.info(f"CPU RAM: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Transformers version: {transformers.__version__}")
    logging.info(f"Python version: {os.popen('python --version').read()}")
    
    os.environ["WANDB_LOG_MODEL"] = "epoch"

    BASE_DIR = base_dir
    if not video_dir:
        video_dir = base_dir
    # av.logging.set_level(av.logging.ERROR)

    # check_cpu_usage(logging)
    model_ckpt = "google/vivit-b-16x2-kinetics400"
    image_processor = VivitImageProcessor.from_pretrained(model_ckpt)
    # check_cpu_usage(logging)
    if torch.cuda.is_available():
        logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(torch.device('cuda'))/1024**3} GB")

    config = VivitConfig.from_pretrained(model_ckpt)
    if not finetune:
        config.num_labels = 1
        config.num_frames = 15

    if not param_search:
        if not finetune:
            model_ft = VivitForVideoClassification(config)
        else:
            # model_ft = VivitForVideoClassification.from_pretrained(
            #     model_ckpt, num_labels=1, ignore_mismatched_sizes=True)
            # check_cpu_usage(logging)
            if torch.cuda.is_available():
                logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(torch.device('cuda'))/1024**3} GB")
            # Freeze all layers except the last one
            # for param in model_ft.parameters():
            #     param.requires_grad = False
            # for param in model_ft.classifier.parameters():
            #     param.requires_grad = True
        num_frames_to_sample = config.num_frames
        sample_rate = image_processor.resample
        # model_ft.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    else:
        num_frames_to_sample = config.num_frames
        sample_rate = image_processor.resample
        model_ft = None

    video_path = os.path.join(video_dir, "videos/")
    train_data = pd.read_json(os.path.join(video_dir, "memento_train_data.json")).sample(frac=sample)
    test_data = pd.read_json(os.path.join(video_dir, "memento_val_data.json")).sample(frac=sample)
    # check_cpu_usage(logging)

    train_data['filename'] = train_data['filename'].apply(lambda x: video_path + x)
    test_data['filename'] = test_data['filename'].apply(lambda x: video_path + x)

    # Search inside the video folder for videos ending in _resampled.mp4 and substitute the original video for it
    # This is done to avoid errors with some videos
    num_of_resampled_videos = 0
    for filename in pd.concat([train_data, test_data], ignore_index=True).filename.values:
        # check_cpu_usage(logging)
        if not filename.endswith("_resampled.mp4") and os.path.exists(os.path.splitext(filename)[0] + "_resampled.mp4"):
            # logging.info(f"Found resampled version of {filename}. Replacing it.")
            filename = os.path.splitext(filename)[0] + "_resampled.mp4"
            num_of_resampled_videos += 1
    logging.info(f"Replaced {num_of_resampled_videos} videos with their resampled versions.")

    # If salient is chosen for frame_sample_strategy, load saliency scores
    if frame_sample_strategy == "salient":
        # Saliency scores is a dataframe that contains one row per frame. group by filename and store the scores for all frames in a list
        logging.info("Loading saliency scores")
        saliency_scores = pd.read_csv(saliency_scores)
        saliency_scores = saliency_scores[['filename', 'frame_path_saliency']]
        saliency_scores["filename"] = saliency_scores["filename"].apply(lambda x: video_path + x)
        saliency_scores = saliency_scores.groupby('filename').frame_path_saliency.apply(list)
        saliency_scores = saliency_scores.apply(lambda x: [float(s) for s in x])

    if frame_sample_strategy == "all-segments":
        train_data = create_segment_database(train_data, 'filename', num_frames_to_sample, 5)

    # check_cpu_usage(logging)

    train_dataset = VideoDataset(
        train_data,
        'filename',
        'mem_score',
        num_frames_to_sample,
        sample_rate,
        video=True,
        processor=image_processor,
        frame_sample_strategy=frame_sample_strategy,
        saliency_scores=saliency_scores
    )
    test_dataset = VideoDataset(
        test_data,
        'filename',
        'mem_score',
        num_frames_to_sample,
        sample_rate,
        video=True,
        processor=image_processor,
        frame_sample_strategy="center" if frame_sample_strategy == "all-segments" else frame_sample_strategy,
        saliency_scores=saliency_scores
    )

    # check_cpu_usage(logging)

    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-memento-{exp_name}"

    # Train using native pytorch
    if method == 'pytorch':
        logging.info("Training using native pytorch")
        metric = evaluate.load("spearmanr")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check_cpu_usage(logging)
        # Log GPU usage
        if torch.cuda.is_available():
            logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(device)/1024**3} GB")
        model_ft.to(device)
        if torch.cuda.is_available():
            logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(device)/1024**3} GB")
        # check_cpu_usage(logging)

        optimizer = torch.optim.AdamW(model_ft.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_loader = train_dataset.load('train', batch_size=batch_size)
        test_loader = test_dataset.load('test', batch_size=batch_size)

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
    
    elif method == 'transformers':
        # Train using transformers Trainer
        # Wandb should report metrics for best epoch, not last epoch
        logging.info("Training using transformers Trainer")
        args = TrainingArguments(
            new_model_name,
            remove_unused_columns=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            auto_find_batch_size=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            #gradient_accumulation_steps=16,
            #tf32=True,
            # optim="sgd",
            lr_scheduler_type="linear",
            # warmup_ratio=0.02,
            logging_steps=10,
            logging_dir=os.path.join(BASE_DIR, "logs", new_model_name),
            load_best_model_at_end=True,
            metric_for_best_model="spearmanr",
            greater_is_better=True,
            report_to="wandb",
            run_name=new_model_name,
        )

        wandb.init(project="training-video-transformers-v3", name=new_model_name, config=args, resume=False)
        trainer = CustomTrainer(
            None,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=image_processor,
            data_collator=collate_fn,
            compute_metrics=compute_spearman,
            model_init=model_init if not finetune else model_init_finetune,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],

        )

        if not param_search:
            # wandb.init(project="huggingface", id="1rgqi7ax", resume="must")
            wandb.define_metric("eval/spearmanr", summary="max")
            if wandb.run.resumed:
                # Find last checkpoint and resume training
                logs_path = os.path.join(BASE_DIR, new_model_name)
                checkpoints = [os.path.join(logs_path, f) for f in os.listdir(logs_path) if re.search("checkpoint", f)]
                last_checkpoint = max(checkpoints, key=os.path.getctime)
                if last_checkpoint is not None:
                    logging.info(f"Resuming training from checkpoint {last_checkpoint}")
                    trainer.train(resume_from_checkpoint=last_checkpoint)
                else:
                    logging.info("No checkpoint found. Starting training from scratch.")
                    trainer.train()
            else:
                trainer.train()
            if args.auto_find_batch_size is True:
                wandb.summary["train/train_batch_size"] = trainer._train_batch_size
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
                metric="eval/spearmanr"
            )
            # Save results
            logging.info(best_trial)

    logging.info("Finished training")
    
    


        

if __name__ == "__main__":
    main()