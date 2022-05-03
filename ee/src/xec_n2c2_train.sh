#!/bin/bash --login

#$ -cwd
# Number of GPU's to be used
#$ -l nvidia_v100=1

# Load the latest CUDA library
# module load libs/cuda

# Enable proxy connection
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128

conda activate dsre-vae
pwd
MODEL=$1 #clinical, base, blue
SPLIT=$2 #split0/ ensemble/
# Applying scispacy
DATA="../../data/${SPLIT}"
#TXT_FILES="${DATA}train/"
#ANN_FILES="${DATA}train/"
#SPACY_FILES="${DATA}spacy/train/"
MODEL_FOLDER="../results/${MODEL}_${SPLIT}"
PRED_FILES="${MODEL_FOLDER}predictions/dev/"
if [ "$3" == "scispacy" ];
then
  echo -e ">> PLM: strating preprocessing ${DATA}\n"
  cd preprocess
  python scispacy.py --datadir "${DATA}train/" --outdir "${DATA}spacy/train/"
  python scispacy.py --datadir "${DATA}dev/" --outdir "${DATA}spacy/dev/"

  echo -e ">> PLM: making data files in "${DATA}spacy/"\n"
  python preprocess_spacy_words.py --txt_files "${DATA}train/"  --ann_files "${DATA}train/"  --spacy_files "${DATA}spacy/train/"
  python preprocess_spacy_words.py --txt_files "${DATA}dev/"  --ann_files "${DATA}dev/"  --spacy_files "${DATA}spacy/dev/"
  cd ..
elif [ "$3" == "make" ];
then
  cd preprocess
  echo -e ">> PLM: making data files in "${DATA}spacy/"\n"
  python preprocess_spacy_words.py --txt_files "${DATA}train/"  --ann_files "${DATA}train/"  --spacy_files "${DATA}spacy/train/"
  python preprocess_spacy_words.py --txt_files "${DATA}dev/"  --ann_files "${DATA}dev/"  --spacy_files "${DATA}spacy/dev/"
  cd ..
else
  echo -e ">> PLM: data files should be in "../data/${SPLIT}spacy"\n"
fi

echo -e ">>PLM: start training with $1 on ${2::-1}\n"
python main.py --config ../configs/local.yaml --mode train --split ${2::-1} --bert ${1}
python eval_script_v3.py "../data/${SPLIT}dev/" $PRED_FILES
