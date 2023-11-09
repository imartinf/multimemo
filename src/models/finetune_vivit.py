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


class VideoDataset(torch.utils.data.Dataset):
    default = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    """
    Dataset class to handle video loading and preprocessing.
    Receives a dictionary in which the key is the path to the video and the value is the label.

    When the dataset is loaded, the videos are processed using an image processor (default is VivitImageProcessor).

    The result is a dictionary with the following keys:
        - pixel_values: tensor of shape (n_frames, n_channels, height, width)
        - label: tensor of shape (1)
    """

    def __init__(
            self,
            data: pd.DataFrame,
            video_col: str,
            label_col: str,
            clip_len: int,
            frame_sample_rate: int,
            video: bool = True,
            processor: transformers.image_processing_utils.BaseImageProcessor = None,
            ) -> None:
        """

        """
        self.videos = []
        self.labels = []
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.video = video
        self.processor = self.default if not processor else processor


        for _, (video_path, label) in data[[video_col, label_col]].iterrows():
            # Append path if self.video is True, otherwise append path without extension
            self.videos.append(os.path.splitext(video_path)[0] if not self.video else video_path)
            self.labels.append(label)

    def _read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        if converted_len >= seg_len:
            # If there are not enough frames, sample uniformly from the entire video and forget about frame_sample_rate
            # This is the case for short videos
            # return np.linspace(0, seg_len, num=clip_len, endpoint=False, dtype=np.int64)
            # If there are not enough frames return a shorter list of indices (later they will be repeated)
            # This is the case for short videos
            start_idx = 0
            end_idx = seg_len
            indices = np.linspace(start_idx, end_idx, num=clip_len)
            indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            return indices
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        assert len(indices) == clip_len, f"Sampled {len(indices)} frames instead of {clip_len}."
        return indices
    
    def _get_item_from_video(self, idx):
        """
        Load video from video file
        """
        # Use yuvj420p format to avoid errors with some videos
        container = av.open(self.videos[idx], format='mp4', mode='r')
        indices = self._sample_frame_indices(self.clip_len, self.frame_sample_rate, container.streams.video[0].frames)
        video = self._read_video_pyav(container, indices)
        # If len(video) < clip_len, repeat the last frame until it reaches clip_len
        if len(video) < self.clip_len:
            video = np.concatenate([np.repeat(video[-1:], self.clip_len - len(video), axis=0), video], axis=0)
            logging.warning(f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame.")
            logging.warning(f"Video shape: {video.shape}")
        assert len(video) == self.clip_len, f"Video {self.videos[idx]} has {len(video)} frames instead of {self.clip_len}."
        label = torch.tensor(self.labels[idx]).float()
        inputs = self.processor(list(video), return_tensors='pt')
        inputs = {k: val.squeeze() for k, val in inputs.items()}
        return inputs, label
    
    def _get_item_from_folder(self, idx):
        """
        Load video from folder with frames
        """
        frames = []
        # Search for a folder named frames inside the video folder
        for root, dirs, files in os.walk(self.videos[idx]):
            if 'frames' in dirs:
                frames_path = os.path.join(root, 'frames')
                break
        # Get all frames in the folder
        for root, _, files in os.walk(frames_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    frames.append(os.path.join(root, file))
        indices = self._sample_frame_indices(self.clip_len, self.frame_sample_rate, len(frames))
        # Load each frame into a numpy array if the index coincides with the filename (frame_0{i}})
        video = []
        for frame in frames:
            # Get the filename from the frame path and extract the index
            frame_idx = int(re.findall(r'\d+', os.path.splitext(os.path.basename(frame))[0])[0])
            if frame_idx in indices:
                img = Image.open(frame)
                # img = np.array(img)
                video.append(img)
        # video = np.stack(video)
        # If len(video) < clip_len, repeat the last frame until it reaches clip_len (as Image, not np.array)
        if len(video) < self.clip_len:
            video = [video[-1]] * (self.clip_len - len(video)) + video
            logging.warning(f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame.")
            logging.warning(f"Video shape: {len(video)}")
        assert len(video) == self.clip_len, f"Video {self.videos[idx]} has {len(video)} frames instead of {self.clip_len}."
        label = torch.tensor(self.labels[idx]).float()
        inputs = self.processor(video, return_tensors='pt')
        inputs = {k: val.squeeze()
                   for k, val in inputs.items()}
        return inputs, label

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._get_item_from_video(idx) if self.video else self._get_item_from_folder(idx)
        

    def load(self, phase: str = 'train', batch_size: int = 32,
                num_workers: int = 0) -> torch.utils.data.DataLoader:
            """Retrieve a DataLoader to ease the pipeline.

            Args:
                phase: Whether it's train or test.
                batch_size: Samples per batch.
                num_workers: Cores to use.

            Returns:
                an iterable torch DataLoader.
            """
            shuffle = True if phase == "train" else False
            return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)


class CustomTrainer(Trainer):
    # Override train method
    def train(self, model_path=None, trial=None, **kwargs):
        self.model = self.model_init(trial)
        # Check if self.model.num_frames equals the appropriate dimension of the data
        # If not, reload the dataset with the correct num_frames
        if self.model.config.num_frames != self.train_dataset[0][0]['pixel_values'].shape[0]:
            logging.info(f"Reloading dataset with num_frames={self.model.config.num_frames} (original num_frames={self.train_dataset[0][0]['pixel_values'].shape[0]})")
            self.train_dataset = VideoDataset(
                pd.DataFrame({'video': self.train_dataset.videos, 'label': self.train_dataset.labels}),
                'video',
                'label',
                self.model.config.num_frames,
                self.train_dataset.frame_sample_rate,
                video=False,
                processor=self.train_dataset.processor
            )
        super().train(resume_from_checkpoint=None, trial=trial)

    # Override evaluate method
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix=None):
        # Check if self.model.num_frames equals the appropriate dimension of the data
        # If not, reload the dataset with the correct num_frames
        if self.model.config.num_frames != self.eval_dataset[0][0]['pixel_values'].shape[0]:
            logging.info(f"Reloading dataset with num_frames={self.model.config.num_frames} (original num_frames={self.eval_dataset[0][0]['pixel_values'].shape[0]})")
            self.eval_dataset = VideoDataset(
                pd.DataFrame({'video': self.eval_dataset.videos, 'label': self.eval_dataset.labels}),
                'video',
                'label',
                self.model.config.num_frames,
                self.eval_dataset.frame_sample_rate,
                video=False,
                processor=self.eval_dataset.processor
            )
        return super().evaluate(eval_dataset, ignore_keys)

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
    config.num_frames = 16
    if trial is None or 'num_frames' not in trial.keys():
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
        "google/vivit-b-16x2-kinetics400", num_labels=1, ignore_mismatched_sizes=True)
    # Freeze all layers except last encoder and regression head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Unfreeze last three transformer blocks
    for param in model.vivit.encoder.layer[-5:].parameters():
        param.requires_grad = True
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
            "num_train_epochs": {"value": 10},
            # "per_device_train_batch_size": {"value": 4},
            "warmup_ratio": {'value': 0.4},
            # Set evaluation batch size equal to training batch size
            # "per_device_eval_batch_size": {"ref": "per_device_train_batch_size"},
            "gradient_accumulation_steps": {"values": [1, 2, 4, 8, 16]},
            # "warmup_ratio": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]},
            "learning_rate": {"value": 1e-04},
            "weight_decay": {"value": 0.0001},
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
@click.option('--num_epochs', type=click.INT, default=1)
@click.option('--sample', type=click.FLOAT, default=1)
def main(base_dir, exp_name, log_dir, video_dir, method, param_search, finetune, learning_rate, batch_size, num_epochs, sample):
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

    os.environ["WANDB_LOG_MODEL"] = "end"

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
        config.num_frames = 16

    if not param_search:
        if not finetune:
            model_ft = VivitForVideoClassification(config)
        else:
            model_ft = VivitForVideoClassification.from_pretrained(
                model_ckpt, num_labels=1, ignore_mismatched_sizes=True)
            # check_cpu_usage(logging)
            if torch.cuda.is_available():
                logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(torch.device('cuda'))/1024**3} GB")
            # Freeze all layers except the last one
            # for param in model_ft.parameters():
            #     param.requires_grad = False
            # for param in model_ft.classifier.parameters():
            #     param.requires_grad = True
        num_frames_to_sample = model_ft.config.num_frames
        sample_rate = image_processor.resample
        model_ft.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
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

    # check_cpu_usage(logging)

    train_dataset = VideoDataset(train_data, 'filename', 'mem_score', num_frames_to_sample, sample_rate, video=True, processor=image_processor)
    test_dataset = VideoDataset(test_data, 'filename', 'mem_score', num_frames_to_sample, sample_rate, video=True, processor=image_processor)

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

        optimizer = torch.optim.AdamW(model_ft.parameters(), lr=5e-5)
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
            num_train_epochs=20,
            auto_find_batch_size=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            #gradient_accumulation_steps=16,
            #tf32=True,
            # Use SGD with momentum as optimizer and cosine scheduler with linear warmup
            # optim="sgd",
            # lr_scheduler_type="cosine",
            # warmup_ratio=0.1,
            logging_steps=10,
            logging_dir=os.path.join(BASE_DIR, "logs", new_model_name),
            load_best_model_at_end=True,
            metric_for_best_model="spearmanr",
            greater_is_better=True,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],

        )

        if not param_search:
            wandb.init(project="huggingface", name=new_model_name, config=args, resume=False)
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