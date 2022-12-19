BASEDIR=$1
TASK=$2

if [ -z "$TASK" ]; then
  echo "Usage: $0 basedir task"
fi


compute_mean() {
  avg=$1
  field=$2
  dev_files=$(find $BASEDIR -name eval_dev.txt -print)
  nums=$(echo $dev_files | xargs -I @ sh -c "grep -A 5 "$TASK" @ | grep "$avg" | cut -d'|' -f $field")
  if [ -z "$nums" ]; then
    echo "No results found for task '$TASK'"
    exit 1
  fi
  sum_expr=$(echo $nums | tr ' ' '+')
  num_files=$(echo $dev_files | wc -w)
  mean_expr=$(printf "scale=4; (%s)/%d" $sum_expr $num_files)
  mean=$(bc <<< "$mean_expr")
  echo "0${mean}"
}

micro_prec=$(compute_mean "MICRO" 3)
micro_rec=$(compute_mean "MICRO" 4)
micro_f1=$(compute_mean "MICRO" 5)

macro_prec=$(compute_mean "MACRO" 3)
macro_rec=$(compute_mean "MACRO" 4)
macro_f1=$(compute_mean "MACRO" 5)

echo "### $TASK"
printf "|       | prec  |  rec  |   f1  |\n"
printf "|-------|-------|-------|-------|\n"
printf "| MICRO | %.3f | %.3f | %.3f |\n" $micro_prec $micro_rec $micro_f1
printf "| MACRO | %.3f | %.3f | %.3f |\n" $macro_prec $macro_rec $macro_f1
