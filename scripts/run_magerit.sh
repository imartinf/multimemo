#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=magerit-finetune-vivit-salient-residual-sampling
#SBATCH --nodes=1
#SBATCH --nodelist=r2n2
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/v530/v530776/multimemo
#SBATCH --output=/home/v530/v530776/multimemo/logs/%j.out
#SBATCH --mail-user=ivan.martinf@upm.es
##------------------------ End job description ------------------------

module purge && module load Python PyTorch

srun python src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-salient-residual-sampling /home/v530/v530776/multimemo/logs \
  --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune true \
  --frame_sample_strategy salient --saliency_scores /home/v530/v530776/multimemo/data/processed/memento_train_test_frames_saliency_residual.csv

# srun accelerate launch src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-all-weights-ddp /home/v530/v530776/multimemo/logs \
#    --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune true --learning_rate 1e-05 --batch_size 1

# srun python scripts/probe_magerit.py