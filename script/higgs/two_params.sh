#!/bin/bash
#SBATCH --job-name=higgs_lfi
#SBATCH --output=out/higgs_lfi_%A_%a.out
#SBATCH --error=err/higgs_lfi_%A_%a.err
#SBATCH --time=00:25:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Create logs directory if it doesn't exist
mkdir -p out
mkdir -p err

# Activate conda environment
source ~/.bashrc
conda activate lfi_benchmark

# Run the Python script with the parameters
python two_params.py