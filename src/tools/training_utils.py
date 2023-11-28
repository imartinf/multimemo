import os
import re
import logging
import numpy as np
import pandas as pd
import torch
import transformers
import av
from PIL import Image

from transformers import Trainer, VivitImageProcessor


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
        return_video_id: bool = False,
    ) -> None:
        """ """
        self.videos = []
        self.labels = []
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.frame_sample_strategy = frame_sample_strategy
        self.video = video
        self.processor = self.default if not processor else processor
        self.return_video_id = return_video_id

        if label_col is None:
            for video_path in data[video_col]:
                # Append path if self.video is True, otherwise append path without extension
                self.videos.append(os.path.splitext(video_path)[0] if not self.video else video_path)
        else:
            for _, (video_path, label) in data[[video_col, label_col]].iterrows():
                # Append path if self.video is True, otherwise append path without extension
                self.videos.append(os.path.splitext(video_path)[0] if not self.video else video_path)
                self.labels.append(label)

    def _read_video_pyav(self, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
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
        """
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
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

    def _sample_center_segment(self, clip_len, seg_len):
        """
        Sample the segment of the video around the center frame.

        Args:
            clip_len (`int`): Total number of frames to sample.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        """

        center_frame_idx = seg_len // 2
        start_idx = max(0, center_frame_idx - clip_len // 2)
        end_idx = min(seg_len, center_frame_idx + clip_len // 2)
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        assert len(indices) == clip_len, f"Sampled {len(indices)} frames instead of {clip_len}."
        return indices

    def _get_item_from_video(self, idx):
        """
        Load video from video file
        """
        # Use yuvj420p format to avoid errors with some videos
        format = self.videos[idx].split(".")[-1]
        try:
            container = av.open(self.videos[idx], format=format, mode="r")
        except Exception as e:
            # If the video is video4592 change the path
            if "video4592" in self.videos[idx]:
                self.videos[idx] = "/mnt/rufus_A/VIDEOMEM/test-set/Videos/video4592.webm"
                container = av.open(self.videos[idx], format=format, mode="r")
            else:
                logging.error(f"Error opening video {self.videos[idx]}: {e}")
                logging.error(f"Skipping video {self.videos[idx]}")
                return None, None
        # Get number of frames
        total_frames = 24 * 7 if format == "webm" else container.streams.video[0].frames
        if self.frame_sample_strategy == "uniform":
            indices = self._sample_frame_indices(self.clip_len, self.frame_sample_rate, total_frames)
        elif self.frame_sample_strategy == "center":
            indices = self._sample_center_segment(self.clip_len, total_frames)
        else:
            raise ValueError(f"Invalid frame sample strategy: {self.frame_sample_strategy}")
        video = self._read_video_pyav(container, indices)
        # If len(video) < clip_len, repeat the last frame until it reaches clip_len
        if len(video) < self.clip_len:
            video = np.concatenate([np.repeat(video[-1:], self.clip_len - len(video), axis=0), video], axis=0)
            logging.warning(
                f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame."
            )
            logging.warning(f"Video shape: {video.shape}")
        assert (
            len(video) == self.clip_len
        ), f"Video {self.videos[idx]} has {len(video)} frames instead of {self.clip_len}."
        label = torch.tensor(self.labels[idx]).float() if len(self.labels) > 0 else None
        inputs = self.processor(list(video), return_tensors="pt")
        inputs = {k: val.squeeze() for k, val in inputs.items()}
        return inputs, label

    def _get_item_from_folder(self, idx):
        """
        Load video from folder with frames
        """
        frames = []
        # Search for a folder named frames inside the video folder
        for root, dirs, files in os.walk(self.videos[idx]):
            if "frames" in dirs:
                frames_path = os.path.join(root, "frames")
                break
        # Get all frames in the folder
        for root, _, files in os.walk(frames_path):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    frames.append(os.path.join(root, file))
        if self.frame_sample_strategy == "uniform":
            indices = self._sample_frame_indices(self.clip_len, self.frame_sample_rate, len(frames))
        elif self.frame_sample_strategy == "center":
            indices = self._sample_center_segment(self.clip_len, len(frames))
        else:
            raise ValueError(f"Invalid frame sample strategy: {self.frame_sample_strategy}")
        # Load each frame into a numpy array if the index coincides with the filename (frame_0{i}})
        video = []
        for frame in frames:
            # Get the filename from the frame path and extract the index
            frame_idx = int(re.findall(r"\d+", os.path.splitext(os.path.basename(frame))[0])[0])
            if frame_idx in indices:
                img = Image.open(frame)
                # img = np.array(img)
                video.append(img)
        # video = np.stack(video)
        # If len(video) < clip_len, repeat the last frame until it reaches clip_len (as Image, not np.array)
        if len(video) < self.clip_len:
            video = [video[-1]] * (self.clip_len - len(video)) + video
            logging.warning(
                f"Video {self.videos[idx]} has less than {self.clip_len} frames. Repeating the last frame."
            )
            logging.warning(f"Video shape: {len(video)}")
        assert (
            len(video) == self.clip_len
        ), f"Video {self.videos[idx]} has {len(video)} frames instead of {self.clip_len}."
        label = torch.tensor(self.labels[idx]).float()
        inputs = self.processor(video, return_tensors="pt")
        inputs = {k: val.squeeze() for k, val in inputs.items()}
        return inputs, label

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int):
        if self.return_video_id:
            return self._get_item_from_video(idx) if self.video else self._get_item_from_folder(idx), self.videos[idx]
        else:
            return self._get_item_from_video(idx) if self.video else self._get_item_from_folder(idx)

    def load(self, phase: str = "train", batch_size: int = 32, num_workers: int = 0) -> torch.utils.data.DataLoader:
        """Retrieve a DataLoader to ease the pipeline.

        Args:
            phase: Whether it's train or test.
            batch_size: Samples per batch.
            num_workers: Cores to use.

        Returns:
            an iterable torch DataLoader.
        """
        shuffle = True if phase == "train" else False
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def get_video_ids(self, batch):
        """Get the video ids from a batch of inputs"""
        video_ids = []
        for video in batch["pixel_values"]:
            video_ids.append(video[0]["filename"])
        return video_ids


class CustomTrainer(Trainer):
    # Override train method
    def train(self, model_path=None, trial=None, **kwargs):
        self.model = self.model_init(trial)
        # Check if self.model.num_frames equals the appropriate dimension of the data
        # If not, reload the dataset with the correct num_frames
        if self.model.config.num_frames != self.train_dataset[0][0]["pixel_values"].shape[0]:
            logging.info(
                f"Reloading dataset with num_frames={self.model.config.num_frames} (original num_frames={self.train_dataset[0][0]['pixel_values'].shape[0]})"
            )
            self.train_dataset = VideoDataset(
                pd.DataFrame({"video": self.train_dataset.videos, "label": self.train_dataset.labels}),
                "video",
                "label",
                self.model.config.num_frames,
                self.train_dataset.frame_sample_rate,
                video=False,
                processor=self.train_dataset.processor,
            )
        super().train(resume_from_checkpoint=None, trial=trial)

    # Override evaluate method
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix=None):
        # Check if self.model.num_frames equals the appropriate dimension of the data
        # If not, reload the dataset with the correct num_frames
        if self.model.config.num_frames != self.eval_dataset[0][0]["pixel_values"].shape[0]:
            logging.info(
                f"Reloading dataset with num_frames={self.model.config.num_frames} (original num_frames={self.eval_dataset[0][0]['pixel_values'].shape[0]})"
            )
            self.eval_dataset = VideoDataset(
                pd.DataFrame({"video": self.eval_dataset.videos, "label": self.eval_dataset.labels}),
                "video",
                "label",
                self.model.config.num_frames,
                self.eval_dataset.frame_sample_rate,
                video=False,
                processor=self.eval_dataset.processor,
            )
        return super().evaluate(eval_dataset, ignore_keys)
