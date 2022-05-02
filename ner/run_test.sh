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
HOME=$PWD
ORG_CORPUS=org_corpus/test
NER_DIR=$HOME/ner
TEST_CORPUS=$NER_DIR/corpus/test

mkdir -p $NER_DIR/corpus
mkdir -p $TEST_CORPUS

echo "Do preprocessing on the test set"
python3 $NER_DIR/src/tokenization/tokenization.py --indir $ORG_CORPUS --outdir $TEST_CORPUS

echo "Predict the test set ..."
cd $NER_DIR

#it is noted that we predict the testing set based on models trained on the organiser's split (data_v3)
for model in $(ls -d experiments/data_v3/*) 
#for model in "baseline"
    do              
        echo "Running $model"  
        python -u src/predict.py --yaml  $model/predict-test.yaml > $model/predict-test.log

        echo "Convert back to the original offsets "
        python src/scripts/postprocess.py $TEST_CORPUS/0 $model/predict-test/ent-last/ent-ann $model/predict-test-org        

    done

#to run ensemble using max voting
mkdir -p test_ensemble
python src/ensemble.py --indir experiments/data_v3 --outdir test_ensemble --type test
cd $HOME