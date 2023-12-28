#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=GAN_training
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=02:40:00
#SBATCH --mem=50G
#SBATCH --export=ALL

# Redirecting stdout and stderr to a file
python /home/bruno/3D-Laision-Seg/GenrativeMethod/model/CT2MRI_3DGAN/main_multiGPU.py

