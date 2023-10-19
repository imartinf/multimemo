#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=probing-magerit
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/v530/v530776/multimemo
#SBATCH --output=/home/v530/v530776/multimemo/logs/%j.out
##------------------------ End job description ------------------------

module purge && module load Python

# srun src/models/finetune_vivit.py --app-param /home/v530/v530776/multimemo prueba_magerit /home/v530/v530776/multimemo/logs \
#    --video_dir /home/v530/v530776/Memento10k --method transformers --param_search true --finetune true --sample 0.01  

srun python scripts/probe_magerit.py