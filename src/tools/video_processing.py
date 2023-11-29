import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.tools import utils


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
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


def sample_center_segment(clip_len, seg_len):
    '''
    Sample the segment of the video around the center frame.

    Args:
        clip_len (`int`): Total number of frames to sample.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    '''

    center_frame_idx = seg_len // 2
    start_idx = max(0, center_frame_idx - clip_len // 2)
    end_idx = min(seg_len, center_frame_idx + clip_len // 2)
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    assert len(indices) == clip_len, f"Sampled {len(indices)} frames instead of {clip_len}."
    return indices


def sample_salient_segment(clip_len, seg_len=None, saliency_sum=None):
    '''
    Sample the segment of the video containing the most salient frames.
    This function must receive an array with the saliency score for each frame (understood as the sum of the saliency for every pixel in the frame).

    Args:
        clip_len (`int`): Total number of frames to sample.
        seg_len (`int`): Maximum allowed index of sample's last frame. Added for compatibility with other sampling functions.
        saliency_sum (`List[float]`): List of saliency scores for each frame.
    '''
    try:
        # Normalize saliency sum to sum 1
        saliency_sum_norm = [s / sum(saliency_sum) for s in saliency_sum]

        # Select the clip_len frame window with highest saliency cumsum
        window_saliency = []
        for i in range(len(saliency_sum_norm) - clip_len):
            window_saliency.append(sum(saliency_sum_norm[i:i+clip_len]))

        # Get the index of the window with highest saliency
        max_saliency_idx = np.argmax(window_saliency)
        # Get the indices of the frames in the window
        start_idx = max_saliency_idx
        end_idx = max_saliency_idx + clip_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        assert len(indices) == clip_len, f"Sampled {len(indices)} frames instead of {clip_len}."
        return indices
    except Exception as e:
        logging.error(f"Error sampling salient segment: {e}")
        logging.error(f"clip_len: {clip_len}")
        logging.error(f"seg_len: {seg_len}")
        logging.error(f"saliency_sum: {saliency_sum}")
        logging.error(f"Resorting to center segment sampling")
        return sample_center_segment(clip_len, seg_len)
    

def collect_all_possible_segments(clip_len, seg_len, frame_overlap):
    '''
    Collect all possible segments of a given length from a video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        seg_len (`int`): Maximum allowed index of sample's last frame.
        frame_overlap (`int`): Number of frames to overlap between consecutive segments.
    Returns:
        segments (`List[List[int]]`): List of all possible segments.
    '''
    segments = []
    start_idx = 0
    end_idx = seg_len
    while end_idx <= seg_len:
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        segments.append(indices)
        start_idx += clip_len - frame_overlap
        end_idx += clip_len - frame_overlap
    return segments

def create_segment_database(dataframe, video_col, clip_len, frame_overlap):
    '''
    Given a pandas DataFrame of videos, create a new DataFrame with all possible segments of a given length.
    Args:
        dataframe (`pandas.DataFrame`): DataFrame with videos.
        clip_len (`int`): Total number of frames to sample.
        seg_len (`int`): Maximum allowed index of sample's last frame.
        frame_overlap (`int`): Number of frames to overlap between consecutive segments.
    '''

    # Create a new DataFrame with all possible segments
    segments = []
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        # Open video
        video_path = row[video_col]
        container = av.open(video_path)
        # Get video length
        seg_len = len(container.streams.video[0])
        # Get all possible segments
        segments.extend(collect_all_possible_segments(clip_len, seg_len, frame_overlap))

    # Append segments to dataframe
    segments_df = dataframe.copy()
    segments_df['segments'] = segments
    segments_df = segments_df.explode('segments')
    segments_df = segments_df.reset_index(drop=True)
    segments_df['segments'] = segments_df['segments'].apply(lambda x: x.tolist())
    segments_df = segments_df.rename(columns={'segments': 'indices'})
    # Create an id that identifies each segment in the context of the video
    segments_df['segment_id'] = segments_df.apply(lambda x: f"{x[video_col]}_{x['indices'][0]}_{x['indices'][-1]}", axis=1)
    return segments_df

    





