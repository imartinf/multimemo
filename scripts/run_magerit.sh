#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=magerit-finetune-vivit-all-weights-ddp
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:2
#SBATCH --chdir=/home/v530/v530776/multimemo
#SBATCH --output=/home/v530/v530776/multimemo/logs/%j.out
#SBATCH --mail-user=ivan.martinf@upm.es
##------------------------ End job description ------------------------

module purge && module load Python PyTorch

# srun python src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-all-weights /home/v530/v530776/multimemo/logs \
#   --video_dir /home/v530/v530776/Memento10k --method transformers --param_search true --finetune true

srun accelerate launch src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit-all-weights-ddp /home/v530/v530776/multimemo/logs \
   --video_dir /home/v530/v530776/Memento10k --method transformers --param_search false --finetune true --learning_rate 1e-05 --batch_size 1

# srun python scripts/probe_magerit.py