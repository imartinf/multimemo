from transformers import VivitImageProcessor, Trainer
import torch

import av
import logging
import numpy as np
import os
import pandas as pd
import re
import torch
import transformers
from PIL import Image

from src.tools.video_processing import sample_frame_indices, sample_center_segment, sample_salient_segment


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
            frame_sample_strategy: str = "uniform",
            video: bool = True,
            processor: transformers.image_processing_utils.BaseImageProcessor = None,
            saliency_scores: pd.DataFrame = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
            ) -> None:
        """

        """
        self.videos = []
        self.labels = []
        self.data = data
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.frame_sample_strategy = frame_sample_strategy
        logging.info(f"Frame sample strategy: {self.frame_sample_strategy}")
        self.video = video
        self.processor = self.default if not processor else processor
        if saliency_scores is not None:
            self.saliency_scores = saliency_scores
        else:
            self.saliency_scores = None
        self.device = device


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
      
    def _get_item(self, idx):
        """
        Load video from video file or folder
        """
        if self.video:
            # Load video from video file
            container = av.open(self.videos[idx], format='mp4', mode='r')
        else:
            # Load video from folder with frames
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
            container = frames
        if self.frame_sample_strategy == "all-segments": 
            if "start_idx" in self.data.columns and "end_idx" in self.data.columns:
                indices = list(range(self.data.loc[idx, "start_idx"], self.data.loc[idx, "end_idx"]))
            else:
                raise ValueError("start_idx and end_idx columns not found in data")
        elif self.frame_sample_strategy == "uniform":
            indices = sample_frame_indices(self.clip_len, self.frame_sample_rate, container.streams.video[0].frames)
        elif self.frame_sample_strategy == "center":
            indices = sample_center_segment(self.clip_len, container.streams.video[0].frames)
        elif self.frame_sample_strategy == "salient":
            # Salient scores is a pandas Series with filename as index and a list of saliency scores for each frame as values
            indices = sample_salient_segment(self.clip_len, container.streams.video[0].frames, self.saliency_scores[self.videos[idx]])
        else:
            raise ValueError(f"Invalid frame sample strategy: {self.frame_sample_strategy}")

        video = self._read_video_pyav(container, indices)

        # If len(video) < clip_len, repeat the last frame until it reaches clip_len
        if len(video) < self.clip_len:
            logging.warning(f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame.")
            logging.warning(f"Video shape: {video.shape}")
            logging.warning(f"Indices: {indices}")
            logging.warning(f"Container frames: {container.streams.video[0].frames}")
            video = np.concatenate([np.repeat(video[-1:], self.clip_len - len(video), axis=0), video], axis=0)

        assert len(video) == self.clip_len, f"Video {self.videos[idx]} has {len(video)} frames instead of {self.clip_len}."

        return video

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # Process video
        video = self._get_item(idx)
        # Process video
        inputs = self.processor(list(video), return_tensors='pt')
        inputs = {k: val.squeeze() for k, val in inputs.items()}
        # Get label
        label = torch.tensor(self.labels[idx]).float()
        return inputs, label
        

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
                processor=self.train_dataset.processor,
                frame_sample_strategy=self.train_dataset.frame_sample_strategy
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
                processor=self.eval_dataset.processor,
                frame_sample_strategy=self.eval_dataset.frame_sample_strategy
            )
        return super().evaluate(eval_dataset, ignore_keys)

######################################################################
#                            DEPREACTED                              #
######################################################################
def _get_item_from_video(self, idx):
        """
        Load video from video file
        """
        # Use yuvj420p format to avoid errors with some videos
        container = av.open(self.videos[idx], format='mp4', mode='r')
        if self.frame_sample_strategy == "uniform":
            indices = sample_frame_indices(self.clip_len, self.frame_sample_rate, container.streams.video[0].frames)
        elif self.frame_sample_strategy == "center":
            indices = sample_center_segment(self.clip_len, container.streams.video[0].frames)
        elif self.frame_sample_strategy == "salient":
            # Salient scores is a pandas Series with filename as index and a list of saliency scores for each frame as values
            indices = sample_salient_segment(self.clip_len, container.streams.video[0].frames, self.saliency_scores[self.videos[idx]])
        else:
            raise ValueError(f"Invalid frame sample strategy: {self.frame_sample_strategy}")
        video = self._read_video_pyav(container, indices)
        # If len(video) < clip_len, repeat the last frame until it reaches clip_len
        if len(video) < self.clip_len:
            video = np.concatenate([np.repeat(video[-1:], self.clip_len - len(video), axis=0), video], axis=0)
            logging.warning(f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame.")
            logging.warning(f"Video shape: {video.shape}")
            logging.warning(f"Indices: {indices}")
            logging.warning(f"Container frames: {container.streams.video[0].frames}")
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
    if self.frame_sample_strategy == "uniform":
        indices = self._sample_frame_indices(self.clip_len, self.frame_sample_rate, len(frames))
    elif self.frame_sample_strategy == "center":
        indices = self._sample_center_segment(self.clip_len, len(frames))
    elif self.frame_sample_strategy == "salient":
        # Salient scores is a pandas Series with filename as index and a list of saliency scores for each frame as values
        indices = self._sample_salient_segment(self.clip_len, len(frames), self.saliency_scores[self.videos[idx]])
    else:
        raise ValueError(f"Invalid frame sample strategy: {self.frame_sample_strategy}")
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
