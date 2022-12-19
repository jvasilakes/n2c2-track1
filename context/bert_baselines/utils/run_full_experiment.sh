CONFIG=$1
QUIET=$2 # --quiet
NSEEDS=2

if [ -z ${CONFIG} ]; then
  echo "Usage: $0 config_file [--quiet]"
  exit 1
fi

# Get the output directory.
# It *should* be the highest numbered version directory + 1
# As long as another instance of run.py train doesn't start in between 
# this command and the run.py line below.
logbasedir=$(grep "^logdir:" $CONFIG | awk '{print $2}')
logsubdir=$(grep "^name:" $CONFIG | awk '{print $2}')
version_dir=$(ls -1d ${logbasedir}/${logsubdir}/version_* | \
              xargs -I @ sh -c 'echo $(basename @)' | \
              sort -t'_' -k2 -n | tail -n 1)
if [ -z ${version_dir} ]; then
  let vnum=1
else
  let vnum=$(echo "$version_dir" | sed -r 's/version_([0-9]+)/\1/')
  let vnum+=1
fi
logdir=${logbasedir}/${logsubdir}/version_${vnum}
echo $logdir
# A poor check for a race condition...
if [ -d "$logdir" ]; then
  echo "Logdir $logdir already exists! Aborting."
  exit 1
fi

# Run training and prediction on the main train/dev split
echo "python run.py $QUIET train $CONFIG"
python run.py $QUIET train $CONFIG

echo "python run.py $QUIET predict ${logdir}/config.yaml"
python run.py $QUIET predict ${logdir}/config.yaml

datadir=$(grep "^data_dir:" $CONFIG | awk '{print $2}')

echo "python ../eval_scripts/eval_context.py \
       --log_file ${logdir}/eval_dev.txt \
       ${logdir}/predictions/n2c2ContextDataset/brat/dev \
       ${datadir}/dev"
python ../eval_scripts/eval_context.py \
       --log_file ${logdir}/eval_dev.txt \
       ${logdir}/predictions/n2c2ContextDataset/brat/dev \
       ${datadir}/dev


# Now create and run the cross validation splits
# datadir is n2c2Track1TrainingData-v3/data_v3, but we want to remove the data_v3
basedatadir=$(dirname $datadir)
bash utils/create_cv_split_configs.sh $logdir $basedatadir
bash utils/run_cv_splits.sh $logdir $QUIET

# Then for each CV split, create and run the random seeds
for cvdir in $(ls -1d ${logdir}/cv_split_runs/*/version_1); do
  echo "Creating random seeds for $cvdir"
  bash utils/create_random_seed_configs.sh $cvdir $NSEEDS
  bash utils/run_random_seeds.sh $cvdir
done
