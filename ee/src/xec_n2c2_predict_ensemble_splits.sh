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
MODEL=$1
#SPLIT=$1 #default, split0
# $3 = dev/
# $4 = ner_predictions/
PRED=$2 #dev_pred/
GOLD_FILES="../data/default/$3" #dev/
PRED_FILES="../results/ensemble_${MODEL}/predictions/test/"

if [ "$4" == "preprocess" ];
then
  cd preprocess
  ## default
  echo -e ">> PLM: preprocessing txt files in ../../data/default/${PRED} with ann in ../../data/default/${PRED}\n"
  python scispacy.py --datadir ../../data/default/${PRED}  --outdir ../../data/default/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/default/${PRED} --ann_files ../../data/default/${PRED} --spacy_files ../../data/default/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/spacy/${PRED::-1}_data.txt \n"
  ## split 0
  echo -e ">> PLM: preprocessing txt files in ../../data/split0/${PRED} with ann in ../../data/split0/${PRED}\n"
  python scispacy.py --datadir ../../data/split0/${PRED}  --outdir ../../data/split0/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/split0/${PRED} --ann_files ../../data/split0/${PRED} --spacy_files ../../data/split0/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/split0/${PRED::-1}_data.txt \n"
  ## split1
  echo -e ">> PLM: preprocessing txt files in ../../data/split1/${PRED} with ann in ../../data/split1/${PRED}\n"
  python scispacy.py --datadir ../../data/split1/${PRED}  --outdir ../../data/split1/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/split1/${PRED} --ann_files ../../data/split1/${PRED} --spacy_files ../../data/split1/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/split1/${PRED::-1}_data.txt \n"
  ## split2
  echo -e ">> PLM: preprocessing txt files in ../../data/split2/${PRED} with ann in ../../data/split2/${PRED}\n"
  python scispacy.py --datadir ../../data/split2/${PRED}  --outdir ../../data/split2/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/split2/${PRED} --ann_files ../../data/split2/${PRED} --spacy_files ../../data/split2/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/split2/${PRED::-1}_data.txt \n"
  ## split3
  echo -e ">> PLM: preprocessing txt files in ../../data/split3/${PRED} with ann in ../../data/split3/${PRED}\n"
  python scispacy.py --datadir ../../data/split3/${PRED}  --outdir ../../data/split3/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/split3/${PRED} --ann_files ../../data/split3/${PRED} --spacy_files ../../data/split3/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/split3/${PRED::-1}_data.txt \n"
  ## split4
  echo -e ">> PLM: preprocessing txt files in ../../data/split4/${PRED} with ann in ../../data/split4/${PRED}\n"
  python scispacy.py --datadir ../../data/split4/${PRED}  --outdir ../../data/split4/spacy/${PRED}
  python preprocess_spacy_words.py --txt_files ../../data/split4/${PRED} --ann_files ../../data/split4/${PRED} --spacy_files ../../data/split4/spacy/${PRED}
  echo -e ">> PLM: data saved in ../../data/default/split4/${PRED::-1}_data.txt \n"

  cd ..
fi
echo -e "PLM: predicting with ${MODEL}, split: default/ data ../data/default/spacy/${PRED::-1}_data.txt > Results in ../results/blue_default/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/default/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_default/"
echo -e "PLM: predicting with ${MODEL}, split: split0/ data ../data/split0/spacy/${PRED::-1}_data.txt > Results in ../results/blue_split0/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/split0/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_split0/"
echo -e "PLM: predicting with ${MODEL}, split: split1/ data ../data/split1/spacy/${PRED::-1}_data.txt > Results in ../results/blue_split1/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/split1/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_split1/"
echo -e "PLM: predicting with ${MODEL}, split: split2/ data ../data/split2/spacy/${PRED::-1}_data.txt > Results in ../results/blue_split2/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/split2/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_split2/"
echo -e "PLM: predicting with ${MODEL}, split: split3/ data ../data/split3/spacy/${PRED::-1}_data.txt > Results in ../results/blue_split3/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/split3/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_split3/"
echo -e "PLM: predicting with ${MODEL}, split: split4/ data ../data/split4/spacy/${PRED::-1}_data.txt > Results in ../results/blue_split4/\n"
python main.py --config ../configs/local.yaml --mode predict --test_path "../data/split4/spacy/${PRED::-1}_data.txt" --bert ${MODEL} --model_folder "../results/${MODEL}_split4/"
python ensemble.py --pred_dir_list ../results/${MODEL}_default/predictions/test/ ../results/${MODEL}_split0/predictions/test/ ../results/${MODEL}_split1/predictions/test/ ../results/${MODEL}_split2/predictions/test/ ../results/${MODEL}_split3/predictions/test/ ../results/${MODEL}_split4/predictions/test/ --ensemble_out ../results/ensemble_${MODEL}/predictions/test/
echo -e ">> Ensemble performance\n"
python eval_script_v3.py $GOLD_FILES $PRED_FILES

