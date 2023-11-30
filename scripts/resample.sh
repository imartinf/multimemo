#!/bin/bash

# Set the desired frame rate as a fraction
FRAMERATE="$1/$2"

# Loop through all video files in the folder
for file in "$3"/*.mp4; do
    # Skip if there is already a resampled version of the video
    if [[ $file == *_resampled.mp4 ]]; then
        continue
    fi
    # Skip if there exists a _resampled.mp4 file
    if [[ -f "${file%.mp4}_resampled.mp4" ]]; then
        continue
    fi
    # Get the current frame rate of the video
    current_framerate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$file")
    # Check if the current frame rate is different from the desired frame rate
    if [ "$current_framerate" != "$FRAMERATE" ]; then
        # Resample the video to the desired frame rate (skip if the resampled version already exists)
        echo "Resampling $file from $current_framerate to $FRAMERATE"        
        ffmpeg -i "$file" -filter:v fps=$FRAMERATE "${file%.mp4}_resampled.mp4"
    fi
done

