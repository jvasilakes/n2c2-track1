#!/bin/sh

#$ -l mem256
# Load any required modulefiles
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128
conda activate dsre-vae
pwd
MODEL=$1 #blue
SPLIT=$2 #ensemble/
# $3 = dev/
# $4 = ner_predictions/
DATA="../../data/${2}"
TXT_FILES="${DATA}${3}"

ANN_FILES=${DATA}${4}
SPACY_FILES="${DATA}spacy/test/" #../../data/ensemble/spacy/test/
SPACY_DATA="../data/${2}spacy/test_data.txt" #../data/ensemble/spacy/test_data.txt
MODEL_FOLDER="../results/${MODEL}_${SPLIT}"
PRED_FILES="${MODEL_FOLDER}predictions/test/"
GOLD_FILES="../data/${2}${3}"
echo -e "PLM: predicting with ${MODEL}, split: ${SPLIT} txt files: ${TXT_FILES}, ann files: ${ANN_FILES}\n Results in ${MODEL_FOLDER}"

if [ "$5" == "preprocess" ];
then
  cd preprocess
  python scispacy.py --datadir $TXT_FILES --outdir $SPACY_FILES
  python preprocess_spacy_words.py --txt_files $TXT_FILES --ann_files $ANN_FILES --spacy_files $SPACY_FILES
  cd ..
fi
python main.py --config ../configs/local.yaml --mode predict --test_path $SPACY_DATA --bert $MODEL --model_folder $MODEL_FOLDER
python eval_script_v3.py $GOLD_FILES $PRED_FILES
