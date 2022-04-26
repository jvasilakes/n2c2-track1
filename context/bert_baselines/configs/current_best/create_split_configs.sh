args=()
for task in "$@"
do
  args+=("${task}.yaml") 
done
if (( ${#args[@]} == 0 ))
then
  args=$(ls -1 *.yaml)
fi
echo "${args[@]}"

current_dir=$(basename $(pwd))
if [[ "$current_dir" != "current_best" ]]; then
  echo "Run this script in the configs/current_best directory. Aborting."
  exit 1
fi

# We'll put the new config files here.
outdir=cv_split_configs
mkdir -p $outdir

for fname in ${args[@]}
do

  # Check if the source file is a symlink, as we'll use this info later.
  if [[ ! -L $fname ]]; then
    echo "$fname is not a symlink. Aborting."
    exit 1
  fi
  # Make sure it links to a valid path.
  if [[ ! -e $fname ]]; then
    echo "$fname is not a valid link. Aborting."
  fi

  # Assume there are 5 splits
  echo "Creating configs for splits and saving them to configs/current_best/cv_split_configs"
  for i in {0..4}; do
    echo "$i"
    # copy the source file with a suffix specifying the split number
    newname=cv_split_configs/${fname/.yaml/${i}.yaml}
    cp $fname $newname
    # get the model name from the config file
    model_name=$(grep "^name:" $newname | awk '{print $2}')
    # get the model version from the symlink of the source file
    model_ver=$(ls -l $fname | grep -Po "version_[0-9]+")
    # Update the config file with the cv_splits and new model name
    # CV split runs get put under a specific model version.
    #python ../../config.py update -f $newname \
    cd ../../
    python -m src.config update -f configs/current_best/$newname \
         -k data_dir "/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/n2c2Track1TrainingData-v3/cv_splits/${i}/" \
         -k sentences_dir "/mnt/iusers01/nactem01/u14498jv/Projects/n2c2-track1/n2c2Track1TrainingData-v3/cv_splits/${i}/segmented" \
         -k name "${model_name}/${model_ver}/cv_split_runs/${i}"
    cd configs/current_best
  done
  # config.py update keeps the originals, but we don't need them.
  rm ${outdir}/*.yaml.orig
done
