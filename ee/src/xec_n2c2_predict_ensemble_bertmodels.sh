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
SPLIT=$1 #default, split0
# $3 = dev/
# $4 = ner_predictions/
DATA="../../data/${1}"
TXT_FILES="${DATA}${2}"

ANN_FILES=${DATA}${3}
SPACY_FILES="${DATA}spacy/test/" #../../data/ensemble/spacy/test/
SPACY_DATA="../data/${1}spacy/test_data.txt" #../data/ensemble/spacy/test_data.txt
MODEL_FOLDER="../results/${MODEL}_${SPLIT}"
GOLD_FILES="../data/${1}${2}"
PRED_FILES="../results/ensemble_${SPLIT}predictions/test/"

if [ "$5" == "preprocess" ];
then
  cd preprocess
  echo -e ">> PLM: preprocessing txt files in ${TXT_FILES} with ann in ${ANN_FILES}\n"
  python scispacy.py --datadir $TXT_FILES --outdir $SPACY_FILES
  python preprocess_spacy_words.py --txt_files $TXT_FILES --ann_files $ANN_FILES --spacy_files $SPACY_FILES
  echo -e ">> PLM: data saved in ${SPACY_DATA} \n"
  cd ..
fi

echo -e "PLM: predicting with blue, split: ${SPLIT} data ${SPACY_DATA} > Results in ../results/blue_${SPLIT}\n"
python main.py --config ../configs/local.yaml --mode predict --test_path $SPACY_DATA --bert blue --model_folder "../results/blue_${SPLIT}"
echo -e "PLM: predicting with clinical, split: ${SPLIT} data ${SPACY_DATA} > Results in ../results/clinical_${SPLIT}\n"
python main.py --config ../configs/local.yaml --mode predict --test_path $SPACY_DATA --bert clinical --model_folder "../results/clinical_${SPLIT}"
python ensemble.py --pred_dir_list ../results/blue_${SPLIT}predictions/test/ ../results/clinical_${SPLIT}predictions/test/ --ensemble_out ../results/ensemble_${SPLIT}predictions/test/
echo -e ">> Blue performance\n"
python eval_script_v3.py $GOLD_FILES "../results/blue_${SPLIT}predictions/test/"
echo -e ">> Clinical performance\n"
python eval_script_v3.py $GOLD_FILES "../results/clinical_${SPLIT}predictions/test/"
echo -e ">> Ensemble performance\n"
python eval_script_v3.py $GOLD_FILES $PRED_FILES