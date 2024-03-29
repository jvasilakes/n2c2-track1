MODEL_DIR=$1
NSEEDS=$2

# We'll put the new config files here.
outdir=${MODEL_DIR}/random_seed_configs
mkdir -p $outdir

echo "Creating configs for ${NSEEDS} random seeds and saving them to ${outdir}"
for i in $(seq 1 ${NSEEDS}); do
  seed=$RANDOM
  echo "${seed}"

  # copy the source file with a suffix specifying the split number
  outfile=${outdir}/seed_${seed}.yaml
  cp ${MODEL_DIR}/config.yaml ${outfile}

  # get the model name from the config file
  model_name=$(grep "^name:" ${outfile} | awk '{print $2}')

  # get the model version from the path of the source file.
  model_ver=$(basename $MODEL_DIR)

  # Update the config file with the new random seed and name for logging.
  python -m src.config update -f ${outfile} \
       -k random_seed ${seed} \
       -k name "${model_name}/${model_ver}/random_seed_runs/${seed}"
  # This creates a backup at ${outfile}.orig, which we don't need.
  rm ${outfile}.orig
done
