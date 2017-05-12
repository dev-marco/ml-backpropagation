ratios='0.5 1.0 10.0'
batch_sizes='1 10 50 inf'
hidden_units='25 50 100'

function big_name_only {
  if [ "${1}" = "1" ]; then
    if [ "${2}" == "latex" ]; then
      echo "Stochastic\ Gradient\ Descent"
    else
      echo "Stochastic Gradient Descent"
    fi
  elif [ "${1}" = "inf" ]; then
    if [ "${2}" == "latex" ]; then
      echo "Gradient\ Descent"
    else
      echo "Gradient Descent"
    fi
  else
    if [ "${2}" == "latex" ]; then
      echo "Batch\ Size"
    else
      echo "Batch Size"
    fi
  fi
}

function big_name {
  if [[ "${1}" = "1" || "${1}" == "inf" ]]; then
    echo "$(big_name_only "$@")"
  else
    echo "$(big_name_only "$@") = ${1}"
  fi
}

function short_name_only {
  if [ "${1}" = "1" ]; then
    echo "SGD"
  elif [ "${1}" = "inf" ]; then
    echo "GD"
  else
    echo "BS"
  fi
}

function short_name {
  if [[ "${1}" = "1" || "${1}" == "inf" ]]; then
    echo $(short_name_only "$@")
  else
    echo "$(short_name_only "$@") = ${1}"
  fi
}

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      labels=$(big_name ${batch})$'\nLearning Rate = '${ratio}$'\nHidden Units = '${hidden}

      ./plot.py experiments/${ratio}-${batch}-${hidden}.txt \
        -labels "${labels}" -validate -folder charts/validations -title-size 10
    done
  done
done

for batch in ${batch_sizes}; do
  files=()
  labels=()

  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("LR = ${ratio}, HU = ${hidden}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts -title "$(big_name ${batch})" -legend-size 6
done

for ratio in ${ratios}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ${batch}), HU = ${hidden}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts -title "Ratio = ${ratio}" -legend-size 6
done

for hidden in ${hidden_units}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for ratio in ${ratios}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ${batch}), LR = ${ratio}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts -title "Hidden Units = ${hidden}" -legend-size 6
done

exit

fixed_ratio=0.5
fixed_batch=10
fixed_hidden=100

echo 'hidden'
mkdir -p charts/hidden

title=$'Hidden Units variation\n$_{(Learning\ Rate = '"${fixed_ratio}"', '"$(big_name ${fixed_batch})"')}$'

./plot.py experiments/${fixed_ratio}-${fixed_batch}-{25,50,100}.txt -labels '25' '50' '100' \
  -folder charts -title "${title}"

labels=(
  $'25 Hidden Units\n$_{('"$(big_name ${fixed_batch} latex)"', Learning\ Rate = '"${fixed_ratio}"')}$'
  $'50 Hidden Units\n$_{('"$(big_name ${fixed_batch} latex)"', Learning\ Rate = '"${fixed_ratio}"')}$'
  $'100 Hidden Units\n$_{('"$(big_name ${fixed_batch} latex)"', Learning\ Rate = '"${fixed_ratio}"')}$'
)

./plot.py experiments/${fixed_ratio}-${fixed_batch}-{25,50,100}.txt \
  -labels "${labels[@]}" -validate -folder charts


echo 'ratio'
mkdir -p charts/ratio

title=$'Learning Rate variation\n$_{('"$(big_name ${fixed_batch} latex)"', Hidden\ Units = '"${fixed_hidden}"')}$'

./plot.py experiments/{0.5,1.0,10.0}-${fixed_batch}-${fixed_hidden}.txt -labels '0.5' '1.0' '10.0' \
  -folder charts -title "${title}"

labels=(
  $'Learning Rate = 0.5\n$_{('"$(big_name ${fixed_batch} latex)"', Hidden\ Units = '"${fixed_hidden}"')}$'
  $'Learning Rate = 1.0\n$_{('"$(big_name ${fixed_batch} latex)"', Hidden\ Units = '"${fixed_hidden}"')}$'
  $'Learning Rate = 10.0\n$_{('"$(big_name ${fixed_batch} latex)"', Hidden\ Units = '"${fixed_hidden}"')}$'
)

./plot.py experiments/{0.5,1.0,10.0}-${fixed_batch}-${fixed_hidden}.txt \
  -labels "${labels[@]}" -validate -folder charts


echo 'batch'
mkdir -p charts/batch

title=$'Batch Size variation\n$_{(Learning\ Rate = '"${fixed_ratio}"', Hidden\ Units = '"${fixed_hidden}"')}$'

./plot.py experiments/${fixed_ratio}-{1,10,50,inf}-${fixed_hidden}.txt \
  -labels $(short_name_only 1) $(short_name_only 10) $(short_name_only 50) $(short_name_only inf) \
  -folder charts -title "${title}"

labels=(
  "$(big_name_only 1)"$'\n$_{(Learning\ Rate = '"${fixed_ratio}"', Hidden\ Units = '"${fixed_hidden}"')}$'
  "$(big_name_only 10)"$'\n$_{(Learning\ Rate = '"${fixed_ratio}"', Hidden\ Units = '"${fixed_hidden}"')}$'
  "$(big_name_only 50)"$'\n$_{(Learning\ Rate = '"${fixed_ratio}"', Hidden\ Units = '"${fixed_hidden}"')}$'
  "$(big_name_only inf)"$'\n$_{(Learning\ Rate = '"${fixed_ratio}"', Hidden\ Units = '"${fixed_hidden}"')}$'
)

./plot.py experiments/${fixed_ratio}-{1,10,50,inf}-${fixed_hidden}.txt \
  -labels "${labels[@]}" -validate -folder charts
