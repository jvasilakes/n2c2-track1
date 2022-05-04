#!/bin/bash --login

#$ -cwd
# Number of GPU's to be used
#$ -l nvidia_v100=1

# Load the latest CUDA library
# module load libs/cuda

# Enable proxy connection
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128
MODEL=$1 # <blue, clinical, base>
SPLIT=$2 # <default, split0-4>
MODE=$3 # <scispacy, make, nothing>
conda activate dsre-vae
pwd
sh xec_n2c2_train.sh $MODEL $SPLIT $MODE
sh xec_n2c2_predict.sh $MODEL $SPLIT dev/ ner_predictions/ preprocess

