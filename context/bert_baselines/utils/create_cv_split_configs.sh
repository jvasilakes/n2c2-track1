MODEL_DIR=$1

# We'll put the new config files here.
outdir=${MODEL_DIR}/cv_split_configs
mkdir -p $outdir

# Assume there are 5 splits
echo "Creating configs for splits and saving them to ${MODEL_DIR}/cv_split_configs"
for i in {0..4}; do
  echo "$i"

  # copy the source file with a suffix specifying the split number
  cp ${MODEL_DIR}/config.yaml ${outdir}/split${i}.yaml

  # get the model name from the config file
  model_name=$(grep "^name:" ${outdir}/split${i}.yaml | awk '{print $2}')

  # get the model version from the symlink of the source file
  model_ver=$(ls -d ${MODEL_DIR} | grep -Po "version_[0-9]+")

  # Update the config file with the cv_splits and new model name
  # CV split runs get put under a specific model version.  
  # this creates a backup at ${outdir}/split${i}.yaml.orig
  python -m src.config update -f ${outdir}/split${i}.yaml \
       -k data_dir "/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/n2c2Track1TrainingData-v3/cv_splits/${i}/" \
       -k sentences_dir "/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/n2c2Track1TrainingData-v3/cv_splits/${i}/segmented" \
       -k name "${model_name}/${model_ver}/cv_split_runs/${i}"
done
rm ${outdir}/*.yaml.orig
