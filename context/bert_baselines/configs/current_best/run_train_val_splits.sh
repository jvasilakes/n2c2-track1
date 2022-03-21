args=()
for task in "$@"
do
  for ((i=0; i<5; ++i))
  do
    args+=("cv_split_configs/${task}${i}.yaml") 
  done
done
if (( ${#args[@]} == 0 ))
then
  args=$(ls -1 cv_split_configs/*.yaml)
fi

current_dir=$(basename $(pwd))
if [[ "$current_dir" != "current_best" ]]; then
  echo "Run this script in the configs/current_best directory. Aborting."
  exit 1
fi

#for config in $(ls -1 cv_split_configs/*.yaml); do
for config in ${args[@]}
do
  echo "============================================"
  echo "$config"
  echo "============================================"
  # Train
  python ../../run.py train $config 
  logbasedir=$(grep "^logdir:" $config | awk '{print $2}')
  logsubdir=$(grep "^name:" $config | awk '{print $2}')
  logdir=${logbasedir}/${logsubdir}/version_1
  logged_config=${logdir}/config.yaml
  if [[ ! -f "${logged_config}" ]]; then
    echo "No logged config file at ${logged_config}. Aborting."
    exit 1
  fi
  # Predict
  python ../../run.py validate ${logged_config} --output_brat
  # Validate and log results to logdir/eval_dev.txt
  datadir=$(grep "^data_dir:" $config | awk '{print $2}')
  python ../../../eval_scripts/eval_context.py \
		--log_file ${logdir}/eval_dev.txt \
		${logdir}/predictions/dev \
		${datadir}/dev
done
