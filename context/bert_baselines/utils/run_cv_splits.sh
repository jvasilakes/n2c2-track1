MODEL_DIR=$1
QUIET=$2 # --quiet

for config in $(ls -1 ${MODEL_DIR}/cv_split_configs/*.yaml); do
  logbasedir=$(grep "^logdir:" $config | awk '{print $2}')
  logsubdir=$(grep "^name:" $config | awk '{print $2}')
  logdir=${logbasedir}/${logsubdir}/version_1
  if [[ -f ${logdir}/eval_dev.txt ]]; then
    echo "Found eval_dev.txt for $(basename $config). Skipping."
    continue
  fi
  echo "============================================"
  echo "$(basename $config)"
  echo "============================================"

  # Train
  python run.py $QUIET train $config

  # Predict
  logged_config=${logdir}/config.yaml
  if [[ ! -f "${logged_config}" ]]; then
    echo "No logged config file at ${logged_config}. Aborting."
    exit 1
  fi

  python run.py $QUIET predict ${logged_config}

  # Validate and log results to logdir/eval_dev.txt
  datadir=$(grep "^data_dir:" $config | awk '{print $2}')
  python ../eval_scripts/eval_context.py \
	--log_file ${logdir}/eval_dev.txt \
	${logdir}/predictions/n2c2ContextDataset/brat/dev \
	${datadir}/dev
done
