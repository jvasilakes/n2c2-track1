help() {
  echo "$0 --task TASK --split DATASPLIT --modeldir MODELDIR --anndir ANNDIR --sentsdir SENTENCESDIR --outdir OUTDIR"
}

if [[ $# -ne 12 ]]; then
  help
  exit 1
fi

while [[ $# > 0 ]]; do
  case $1 in 
  --task)     TASK=$2
              shift
              ;;
  --split)    DATASPLIT=$2
              shift
              ;;
  --modeldir) MODELDIR=$2
              shift
              ;;
  --anndir)   ANNDIR=$2
              shift
              ;;
  --sentsdir) SENTENCESDIR=$2
              shift
              ;;
  --outdir)   OUTDIR=$2
              shift
              ;;
  --help)     help
              exit 1
              ;;
  *)          echo "Unsupported argument $1"
              exit 1;;
  esac
  shift
done

echo TASK: $TASK
echo SPLIT: $DATASPLIT
echo MODELDIR $MODELDIR
echo ANNDIR: $ANNDIR
echo SENTENCESDIR: $SENTENCESDIR
echo OUTDIR: $OUTDIR

# Find all model subdirectories
model_dirs=$(find -L ${MODELDIR} -name config.yaml -exec dirname {} \; | grep -v "random_seed_runs")

# Run prediction on the specified data using each model.
for model_dir in ${model_dirs}; do
  echo "========================================================="
  echo "Predicting using model at"
  echo "${model_dir}"
  echo "========================================================="
  python run.py predict ${model_dir}/config.yaml \
                        --datasplit ${DATASPLIT} \
                        --datadir ${ANNDIR} \
                        --sentences_dir ${SENTENCESDIR}
  echo "python run.py predict ${model_dir}/config.yaml --datasplit ${DATASPLIT} --datadir ${ANNDIR} --sentences_dir ${SENTENCESDIR}"
done
 

# Create a unique output directory
mkdir -p ${OUTDIR}/${DATASPLIT}/${TASK}
version=$(find ${OUTDIR}/${DATASPLIT}/${TASK}/ -name 'version_*' -print | wc -l)
let version=${version}+1

# Ensemble the predictions using max voting.
model_dirs_cmdline=$(echo ${model_dirs} | xargs)
python run_ensemble.py --dataset n2c2ContextDataset \
                       --datasplit test --task ${TASK} \
                       --model_dirs ${model_dirs_cmdline} \
                       --outdir ${OUTDIR}/${DATASPLIT}/${TASK}/version_${version}/predictions/
echo "Ensembled predictions saved to ${OUTDIR}/${DATASPLIT}/${TASK}/version_${version}/"
echo "${model_dirs}" > ${OUTDIR}/${DATASPLIT}/${TASK}/version_${version}/ensembled_models.txt
