BASEDIR=$1
TASK=$2
VERBOSE=$3  # --verbose

if [[ "$#" -ne 2 && "$#" -ne 3 ]]; then
  echo "Usage: $0 basedir task [--verbose]"
  exit 1
fi

if [ "$VERBOSE" != "--verbose" ]; then
  VERBOSE=
fi

get_nums() {
  avg=$1
  field=$2
  dev_files=$(find $BASEDIR -name eval_dev.txt -print)
  nums=$(echo $dev_files | xargs -I @ sh -c "grep -A 5 "$TASK" @ | grep "$avg" | cut -d'|' -f $field")
  echo "$nums" | tr '\n' ' '
}

compute_mean() {
  avg=$1
  field=$2
  nums=$(get_nums "$avg" "$field")
  if [ -z "$nums" ]; then
    echo "No results found for task '$TASK'"
    exit 1
  fi
  sum_expr=$(echo $nums | tr ' ' '+')
  num_files=$(echo $nums | wc -w)
  mean_expr=$(printf "scale=4; (%s)/%d" $sum_expr $num_files)
  mean=$(bc <<< "$mean_expr")
  echo "0${mean}"
}

harmonic_mean() {
  a=$1
  b=$2
  harm_expr=$(printf "scale=4; (2 * %f * %f)/(%f + %f)" $a $b $a $b)
  harm=$(bc <<< "$harm_expr")
  echo "0${harm}"
}


micro_prec=$(compute_mean "MICRO" 3)
micro_rec=$(compute_mean "MICRO" 4)
#micro_f1=$(compute_mean "MICRO" 5)
micro_f1=$(harmonic_mean $micro_prec $micro_rec)

macro_prec=$(compute_mean "MACRO" 3)
macro_rec=$(compute_mean "MACRO" 4)
#macro_f1=$(compute_mean "MACRO" 5)
macro_f1=$(harmonic_mean $macro_prec $macro_rec)

echo "### $TASK"
numfiles=$(find $BASEDIR -name eval_dev.txt -print | wc -l)
echo "Summarizing $numfiles files."

if [ ! -z "$VERBOSE" ]; then
  echo "MICRO"
  echo "  P: $(get_nums "MICRO" 3)"
  echo "  R: $(get_nums "MICRO" 4)"
  echo "  F: $(get_nums "MICRO" 5)"
  echo "MACRO"
  echo "  P: $(get_nums "MACRO" 3)"
  echo "  R: $(get_nums "MACRO" 4)"
  echo "  F: $(get_nums "MACRO" 5)"
fi

printf "|       | prec  |  rec  |   f1  |\n"
printf "|-------|-------|-------|-------|\n"
printf "| MICRO | %.3f | %.3f | %.3f |\n" $micro_prec $micro_rec $micro_f1
printf "| MACRO | %.3f | %.3f | %.3f |\n" $macro_prec $macro_rec $macro_f1
