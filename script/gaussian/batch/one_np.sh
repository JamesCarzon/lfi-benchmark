#!/bin/bash
#SBATCH --job-name=gaussian_lfi
#SBATCH --output=out/gaussian_lfi_%A_%a.out
#SBATCH --error=err/gaussian_lfi_%A_%a.err
#SBATCH --array=1-3
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Create logs directory if it doesn't exist
mkdir -p out
mkdir -p err

# Activate conda environment
source ~/.bashrc
conda activate lfi_benchmark

# Read the parameter file
PARAM_FILE="array_params.txt"
LINE_NUMBER=$SLURM_ARRAY_TASK_ID

# Extract parameters for this job
PARAMS=$(sed -n "${LINE_NUMBER}p" $PARAM_FILE)

# Parse parameters (assuming format: n sigma val_split)
read -r N SIGMA VAL_SPLIT <<< "$PARAMS"

echo "Job ${SLURM_ARRAY_TASK_ID}: Running with N=${N}, sigma=${SIGMA}, val_split=${VAL_SPLIT}"

# Run the Python script with the parameters
python ../dev/one_np.py --n $N --sigma $SIGMA --val_split $VAL_SPLIT

echo "Job ${SLURM_ARRAY_TASK_ID} completed"