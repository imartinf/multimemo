#!/bin/zsh

# CENTER SEGMENT
echo "CENTER SEGMENT (DEV-SET)"
while true; do
    python /mnt/rufus_A/multimemo/test/evaluate_mediaeval.py \
        --model_folder_path /mnt/rufus_A/models/vivit-finetune-memento-center-segment-2 \
        --data_path /mnt/rufus_A/VIDEOMEM/dev-set/ground-truth_dev-set.csv \
        --video_folder_path /mnt/rufus_A/VIDEOMEM/dev-set/Videos/ \
        --save_path /mnt/rufus_A/multimemo/runs/vivit-finetune-memento-center-segment-2/me23mem-THAU-UPM-subtask1-runfViViTCenterSegmentDevSet.csv \
    # Check if the script was killed
    if [ $? -eq 137 ]; then
        # Sleep 10 seconds and relaunch
        sleep 10
        echo "Script was killed. Relaunching..."
    else
        break
    fi
done

# SALIENT SEGMENT
echo "SALIENT SEGMENT (DEV-SET)"
while true; do
    python /mnt/rufus_A/multimemo/test/evaluate_mediaeval.py \
        --model_folder_path /mnt/rufus_A/models/vivit-finetune-memento-salient-sampling \
        --data_path /mnt/rufus_A/VIDEOMEM/dev-set/ground-truth_dev-set.csv \
        --video_folder_path /mnt/rufus_A/VIDEOMEM/dev-set/Videos/ \
        --save_path /mnt/rufus_A/multimemo/runs/vivit-finetune-memento-salient-sampling/me23mem-THAU-UPM-subtask1-runfViViTSalientSegmentDevSet.csv \
    # Check if the script was killed
    if [ $? -eq 137 ]; then
        # Sleep 10 seconds and relaunch
        sleep 10
        echo "Script was killed. Relaunching..."
    else
        break
    fi
done

# UNIFORM SAMPLING
echo "UNIFORM SAMPLING (DEV-SET)"
while true; do
    python /mnt/rufus_A/multimemo/test/evaluate_mediaeval.py \
        --model_folder_path /mnt/rufus_A/models/vivit-finetune-memento-uniform-sampling \
        --data_path /mnt/rufus_A/VIDEOMEM/dev-set/ground-truth_dev-set.csv \
        --video_folder_path /mnt/rufus_A/VIDEOMEM/dev-set/Videos/ \
        --save_path /mnt/rufus_A/multimemo/runs/vivit-finetune-memento-uniform-sampling/me23mem-THAU-UPM-subtask1-runfViViTUniformSamplingDevSet.csv \
    # Check if the script was killed
    if [ $? -eq 137 ]; then
        # Sleep 10 seconds and relaunch
        sleep 10
        echo "Script was killed. Relaunching..."
    else
        break
    fi
done

# SALIENT RESIDUAL SAMPLING
echo "SALIENT RESIDUAL SAMPLING (DEV-SET)"
while true; do
    python /mnt/rufus_A/multimemo/test/evaluate_mediaeval.py \
        --model_folder_path /mnt/rufus_A/models/vivit-finetune-memento-salient-residual-sampling \
        --data_path /mnt/rufus_A/VIDEOMEM/dev-set/ground-truth_dev-set.csv \
        --video_folder_path /mnt/rufus_A/VIDEOMEM/dev-set/Videos/ \
        --save_path /mnt/rufus_A/multimemo/runs/vivit-finetune-memento-salient-residual-sampling/me23mem-THAU-UPM-subtask1-runfViViTSalientResidualSamplingDevSet.csv \
    # Check if the script was killed
    if [ $? -eq 137 ]; then
        # Sleep 10 seconds and relaunch
        sleep 10
        echo "Script was killed. Relaunching..."
    else
        break
    fi
done