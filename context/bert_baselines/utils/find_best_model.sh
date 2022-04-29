TASK=$1
AVG=$(echo $2 | tr '[:lower:]' '[:upper:]')

if [[ -z "$3" ]]; then
  N=1
else
  N=$3
fi

echo "$TASK: $AVG"
find ~/scratch/n2c2_track1/context/bert_baseline_logs/ -name eval_dev.txt -print | grep -i "$TASK" | grep -v "cv_split" | xargs grep -i -A 5 "## $TASK" | grep "$AVG" | sort -t'|' -k5 -r | head -n $N
