#!/bin/bash -e
################################################################################ 
# Creates and runs a basic tensorflow example on GPU nodes. The tensorflow conda 
# environment is created 
# 
# Run with: 
# 
#   $ sbatch tensorflow-example.slurm 
# 
################################################################################
 
#SBATCH --account=glaz.p_radar          # Your account
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=a100:1                 # allocate n (out of 4) A100 GPUs; to get n (out of 2) A40 GPUs use --gpus=a40:2
#SBATCH --hint=nomultithread
#SBATCH --qos=12h
#SBATCH --job-name=hameed_job
#SBATCH --output=out_%x.%j.log
#SBATCH	--error=out_%x.%j.log

 
module load conda
module load gcc
module load cuda

# Go to the work location
PROJECT_BASE=/albedo/work/user/hmoqadam/SEGMENTATION_METHOD_RADDLEHAM/

# Ensure we always re-build the environment from scratch for right now...
if [ -d ${PROJECT_BASE}/conda_env ]; then
    rm -rf ${PROJECT_BASE}/conda_env
fi
mamba env create -f ${PROJECT_BASE}/environment.yml -p ${PROJECT_BASE}/conda_env/tensorflow-gpu

cd $PROJECT_BASE
conda activate ${PROJECT_BASE}/conda_env/tensorflow-gpu
time python main_tuning.py






