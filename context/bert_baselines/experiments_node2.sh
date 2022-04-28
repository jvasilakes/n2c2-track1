#!/bin/bash --login
#$ -N micros
#$ -o /mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs/avg_micro_f1_node2_output.txt
#$ -e /mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs/avg_micro_f1_node2_error.txt

# One GPU
#$ -l v100

# The -pe line is optional. Number of CPU cores N can be 2..32
# (max 8 per GPU). Will be a serial job if this line is missing.
#$ -pe smp.pe 8

module load libs/cuda
conda activate n2c2


CONTEXT_HOME=/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/context/bert_baselines
BASELOGDIR=/mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs
GOLD_DATADIR=/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/n2c2Track1TrainingData-v3/data_v3

cd ${CONTEXT_HOME}

run_micro() {
  # e.g. bluebert/mdl/n2c2action_i2b2event/version_1
  model_path=$1

  # bluebert/mdl/n2c2action_i2b2event/
  new_config_dir="configs/$(dirname ${model_path})_micro"
  # version_1_micro.yaml
  config_name=$(basename ${model_path})
  mkdir -p ${CONTEXT_HOME}/${new_config_dir}
  cp ${BASELOGDIR}/${model_path}/config.yaml ${CONTEXT_HOME}/${new_config_dir}/${config_name}.yaml

  python -m src.config update \
    -k description "Same as ${config_name} but tuning to Micro F1 score." \
    -k monitor avg_micro_f1 \
    -k logdir ${BASELOGDIR}/avg_micro_f1 \
    -f ${CONTEXT_HOME}/${new_config_dir}/${config_name}.yaml
  rm ${CONTEXT_HOME}/${new_config_dir}/${config_name}.yaml.orig

  python run.py --quiet train ${CONTEXT_HOME}/${new_config_dir}/${config_name}.yaml

  # Predict and evaluate on dev
  # xargs trims whitespace
  exp_name=$(grep "^name:" ${CONTEXT_HOME}/${new_config_dir}/${config_name}.yaml | awk -F':' '{print $2}' | xargs)
  python run.py --quiet predict ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/config.yaml
  python ../eval_scripts/eval_context.py --log_file ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/eval_dev.txt \
                                         ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/predictions/n2c2ContextDataset/brat/dev/ \
                                         ${GOLD_DATADIR}/dev/
  
  bash utils/create_cv_split_configs.sh ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/
  bash utils/run_cv_splits.sh ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/ --quiet

  echo "==============================================================="
  echo "Run completed: ${BASELOGDIR}/avg_micro_f1/${exp_name}/version_1/"
  echo "==============================================================="
}



# Negation
## clinical_bert/mtl/action_negation/version_3/ retuned to avg_micro_f1
model=clinical_bert/mtl/action_negation/version_3/
run_micro "$model"

# Temporality
## clinical_bert/mdl/n2c2_i2b2_temporality/version_3/ retuned to avg_micro_f1
model=clinical_bert/mdl/n2c2_i2b2_temporality/version_3/
run_micro "$model"

## clinical_bert/temporality_entity_marked/version_3 retuned to avg_micro_f1
model=clinical_bert/temporality_entity_marked/version_3
run_micro "$model"
