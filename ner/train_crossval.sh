#!/bin/bash --login

#$ -cwd

# Number of GPU's to be used
#$ -l nvidia_v100=1

# Number of CPU cores to be used
# 

# echo "Job is using $NGPUS GPU's with ID's $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core's"

# Load the latest CUDA library
# module load libs/cuda

# Enable proxy connection
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128

# Create conda env

#conda update -n base -c defaults conda
#yes | conda create -n deepEM-vae python=3.8.3 
# conda activate n2c2

# Install packages
#yes | conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
#pwd
#pip install -r requirements.txt

# Training model
# HOME=$PWD
ORG_CORPUS=n2c2Track1TrainingData-v3/cv_splits
# NER_DIR=$HOME/ner
TRAIN_CORPUS=ner/corpus

EXP_DIR=ner/experiments/

# for i in "0" "1" "2" "3" "4" "data-v3"
for i in "data-v3"
do
    folderi=$EXP_DIR$i    
    for model in "baseline"
    do
        #remove old results
        rm -r $folderi/$model/*_loss
        rm $folderi/$model/events*
        rm $folderi/$model/joint/*
        rm $folderi/$model/model/*

        echo "Running $folderi/$model"
                
        python -u ner/src/main.py --yaml $folderi/$model/train-ner-vae.yaml > $folderi/$model/train-ner-vae.log

        echo "Predict and evaluate on the development set"
        python -u ner/src/predict.py --yaml  $folderi/$model/predict-dev.yaml > $folderi/$model/predict-dev.log

        echo "Convert back to the original offsets and do evaluation"
        python ner/src/scripts/postprocess.py $TRAIN_CORPUS/$i/dev $folderi/$model/predict-dev/ent-last/ent-ann $folderi/$model/predict-dev-org

        python ner/eval_scripts/n2c2.py $ORG_CORPUS/$i/dev $folderi/$model/predict-dev-org > $folderi/$model/result_org.txt

    done
done
