#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=magerit-vivit-memento-scratch-gabi-params
#SBATCH --nodes=1
#SBATCH --nodelist=r2n1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/v530/v530776/multimemo
#SBATCH --output=/home/v530/v530776/multimemo/logs/%j.out
#SBATCH --mail-user=ivan.martinf@upm.es
##------------------------ End job description ------------------------

module purge && module load Python PyTorch

# srun python src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-uniform-sampling-2 /home/v530/v530776/multimemo/logs \
#   --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune true \
#   --frame_sample_strategy uniform --saliency_scores /home/v530/v530776/multimemo/data/processed/memento_train_test_frames_saliency_residual.csv

srun python src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-vivit-memento-scratch-gabi-params /home/v530/v530776/multimemo/logs \
  --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune false \
  --frame_sample_strategy center --num_epochs 50 --learning_rate 0.0001

# srun accelerate launch src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-all-weights-ddp /home/v530/v530776/multimemo/logs \
#    --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune true --learning_rate 1e-05 --batch_size 1

# srun python scripts/probe_magerit.py