#!/bin/bash --login

#$ -cwd
# Number of GPU's to be used
#$ -l nvidia_v100=1

# Load the latest CUDA library
# module load libs/cuda

# Enable proxy connection
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128
MODE=$2 #scispacy, make, nothing
conda activate dsre-vae
pwd


if [ "$1" == "train" ];
then
  sh xec_n2c2_train.sh blue default/ $MODE
  sh xec_n2c2_train.sh blue split0/ $MODE
  sh xec_n2c2_train.sh blue split1/ $MODE
  sh xec_n2c2_train.sh blue split2/ $MODE
  sh xec_n2c2_train.sh blue split3/ $MODE
  sh xec_n2c2_train.sh blue split4/ $MODE
  ## clinical
  sh xec_n2c2_train.sh clinical default/ $MODE
  sh xec_n2c2_train.sh clinical split0/ $MODE
  sh xec_n2c2_train.sh clinical split1/ $MODE
  sh xec_n2c2_train.sh clinical split2/ $MODE
  sh xec_n2c2_train.sh clinical split3/ $MODE
  sh xec_n2c2_train.sh clinical split4/ $MODE
elif [ "$1" == "test" ];
then
  sh xec_n2c2_predict.sh blue default/ ner_predictions/ preprocess
  sh xec_n2c2_predict.sh blue split0/ ner_predictions/ preprocess
  sh xec_n2c2_predict.sh blue split1/ ner_predictions/ preprocess
  sh xec_n2c2_predict.sh blue split2/ ner_predictions/ preprocess
  sh xec_n2c2_predict.sh blue split3/ ner_predictions/ preprocess
  sh xec_n2c2_predict.sh blue split4/ ner_predictions/ preprocess
fi