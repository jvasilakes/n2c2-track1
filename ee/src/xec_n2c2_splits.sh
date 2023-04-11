#!/bin/bash --login

MODEL=$1 # <blue, clinical, base>
#SPLIT=$2 # <default, split0-4>
MODE=$2 # <scispacy, make, run>
# $3 <only_typed_verbs, no_verb_categories, nothing>
conda activate n2c2_panos
pwd
for SPLIT in default/ split0/ split1/ split2/ split3/ split4/
do
  sh xec_n2c2_train.sh $MODEL $SPLIT $MODE $3 > "out_${SPLIT::-1}_${MODEL}_${3}.txt"
done


