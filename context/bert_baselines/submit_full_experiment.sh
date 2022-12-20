usage() {
  echo "Usage: $(basename $0) experiment_name config_file"
  exit 1
}

if [ "$#" -ne 2 ]; then
  usage
fi

NAME=$1
CONFIG=$2
NSEEDS=4

if [ ! -f "${CONFIG}" ]; then
  echo "Couldn't locate config file '${CONFIG}'. Aborting."
  exit 1
fi

outdir=${PWD}/batch_jobs/${NAME}
# Make sure this run is unique
if [ -d "${outdir}" ]; then
  echo "Found existing run at '${outdir}'. Aborting."
  exit 1
fi
mkdir -p $outdir

# Save this precise config file just to make sure
# the script runs using the correct one.
cp $CONFIG ${outdir}/config.yaml

# Generate a submission script
echo "#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=3:00:00
#$ -o ${outdir}/output.log
#$ -e ${outdir}/error.log
#$ -cwd
#$ -N ${NAME}

module load gcc/9.3.0 python/3.10/3.10.4 cuda/11.3/11.3.1 cudnn/8.2/8.2.4
source /home/ace14853wv/venv/n2c2/bin/activate

bash utils/run_full_experiment.sh ${outdir}/config.yaml --quiet $NSEEDS" > ${outdir}/submit.sh

# Submit it
qsub -g gae50975 ${outdir}/submit.sh
