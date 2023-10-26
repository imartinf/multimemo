#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=magerit-finetune-vivit
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/v530/v530776/multimemo
#SBATCH --output=/home/v530/v530776/multimemo/logs/%j.out
#SBATCH --mail-user=ivan.martinf@upm.es
##------------------------ End job description ------------------------

module purge && module load Python PyTorch

srun python src/models/finetune_vivit.py /home/v530/v530776/multimemo magerit-finetune-vivit /home/v530/v530776/multimemo/logs \
   --video_dir /home/v530/v530776/Memento10k --method transformers --param_search true --finetune true

# srun python scripts/probe_magerit.py