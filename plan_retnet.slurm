#!/bin/bash -l
#SBATCH --job-name=plan_retnet
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="m.planas.lisbona@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gpus=1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change

module purge

echo "## Starting GPU test on $HOSTNAME"
echo "## Loading module"

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"


module load Python/3.8.6-GCCcore-11.3.0
conda activate /data1/s3686140/.conda/envs/trajectory


echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python scripts/plan.py --dataset bullet-halfcheetah-medium-v0 --model retnet
echo "[$SHELL] ## Script finished"

