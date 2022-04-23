#!/bin/sh

#$ -l mem256
# Load any required modulefiles
# module load tools/env/proxy2        # Uses http://proxy.man.ac.uk:3128
conda activate dsre-vae
pwd

echo -e "PLM: start training with $1\n"

#OUT="re_${1}_dev.txt"
#ERR="re_${1}_dev.err"
#
#python train_relation.py --yaml ../config_pipeline.yaml --data $1 --scenario $2 --dataset $3  > $OUT 2> $ERR
#
## $1: clinical, base, blue
## $2:
# Applying scispacy
python main.py --config ../configs/local.yaml --mode train --split default --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split split0 --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split split1 --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split split2 --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split split3 --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split split4 --bert ${1}
python main.py --config ../configs/local.yaml --mode train --split ensemble --bert ${1}