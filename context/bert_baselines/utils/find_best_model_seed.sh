TASK=$1
AVG=$(echo $2 | tr '[:lower:]' '[:upper:]')

echo "$TASK: $AVG"

# Get the models with random seed evaluations
SEED_MODELS="$(find ~/scratch/n2c2_track1/context/bert_baseline_logs/ -name random_seed_runs -exec dirname {} \; | grep -i "$TASK")"

results=()
for model in $SEED_MODELS; do
  # Compute the average of each metric (precision, recall, F1) over the random seeds
  res="$(grep -A 5 -i "## ${TASK}" $model/random_seed_runs/*/version_1/eval_dev.txt | grep "$AVG" | awk -F'|' '{psum+=$3; rsum+=$4; fsum+=$5; total+=1} END{printf "%s | %s | %s \n", psum/total, rsum/total, fsum/total}')"
  # Save the result
  if [[ "${res}" != "" ]]; then
    results+=("${model}: ${res}\n") 
  fi
done

# Sort the results by F1 score
IFS=$'\n' sorted=($(sort -r -t '|' -k 3 <<<"${results[*]}")); unset IFS
printf "%s\n" "${sorted[@]}"
