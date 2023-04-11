#!/bin/bash

#your command lines to run your model
MODEL=$1 #blue
SPLIT=$2 #default/, split0-4/
# $3 = #test_pred/
# $4 = scispasy, run
DATA="../../data/${2}"
TXT_FILES="${DATA}${3}"
ANN_FILES=${DATA}${3}
SPACY_FILES="${DATA}spacy/$3" #../../data/split0/spacy/test_pred/
SPACY_DATA="../data/${2}spacy/${3::-1}_data.txt" #../data/ensemble/spacy/test_data.txt
MODEL_FOLDER="../results/${MODEL}_${SPLIT}"
PRED_FILES="${MODEL_FOLDER}predictions/test/"
GOLD_FILES="../data/${2}test/"


if [ "$4" == "scispacy" ];
then
  cd preprocess
  echo -e ">> PLM: preprocessing txt files in ${TXT_FILES} with ann in ${ANN_FILES}\n"
  python scispacy.py --datadir $TXT_FILES --outdir $SPACY_FILES
  python preprocess_spacy_words.py --txt_files $TXT_FILES --ann_files $ANN_FILES --spacy_files $SPACY_FILES
  echo -e ">> PLM: data saved in ${SPACY_DATA} \n"
  cd ..
fi
echo -e "PLM: predicting with ${MODEL}, split: ${SPLIT} data ${SPACY_DATA} > Results in ${MODEL_FOLDER}\n"
python main.py --config ../configs/local.yaml --mode predict --test_path $SPACY_DATA --bert $MODEL --model_folder $MODEL_FOLDER --approach types
python eval_script_v3.py $GOLD_FILES $PRED_FILES
