#!/bin/bash

HOME=$PWD
ORG_CORPUS=$HOME/n2c2Track1TrainingData-v3/cv_splits
NER_DIR=$HOME/ner
TRAIN_CORPUS=$NER_DIR/corpus

mkdir -p $TRAIN_CORPUS

#process the data, tokenisation
#note: each folder should contain sub-folders such as train, dev, test
for i in {0..4}
do
    echo "Processing $ORG_CORPUS/$i"
    python3 $HOME/ner/src/tokenization/tokenization.py --indir $ORG_CORPUS/$i --outdir $TRAIN_CORPUS/$i
done

####Download language models
echo "Download RoBERTa-large-PM-M3-Voc"
mkdir -p $NER_DIR/pre-trained-model/
cd $NER_DIR/pre-trained-model//
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -xzvf RoBERTa-large-PM-M3-Voc-hf.tar.gz
mv RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf/ roberta-pm-m3
rm -r RoBERTa-large-PM-M3-Voc
rm RoBERTa-large-PM-M3-Voc-hf.tar.gz

cd $HOME



