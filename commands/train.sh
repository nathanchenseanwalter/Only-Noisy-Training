#!/bin/bash
#SBATCH --partition=mig
#SBATCH --job-name=train_only-noisy-training         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
source .venv/bin/activate

python train.py