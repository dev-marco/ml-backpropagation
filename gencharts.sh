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

mkdir -p charts/var3

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      labels=$(big_name ${batch})$'\nLearning Rate = '${ratio}$'\nHidden Units = '${hidden}

      ./plot.py experiments/${ratio}-${batch}-${hidden}.txt \
        -labels "${labels}" -validate -folder charts/var3 -title-size 10
    done
  done
done

mkdir -p charts/var2/batch

for batch in ${batch_sizes}; do
  files=()
  labels=()

  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("LR = ${ratio}, HU = ${hidden}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts/var2/batch -title "$(big_name ${batch})" -legend-size 6
done

mkdir -p charts/var2/ratio

for ratio in ${ratios}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ${batch}), HU = ${hidden}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts/var2/ratio -title "Ratio = ${ratio}" -legend-size 6
done

mkdir -p charts/var2/hidden

for hidden in ${hidden_units}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for ratio in ${ratios}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ${batch}), LR = ${ratio}")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder charts/var2/hidden -title "Hidden Units = ${hidden}" -legend-size 6
done

echo 'hidden'
mkdir -p charts/var1/hidden

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    title=$'Hidden Units variation\n$_{('"$(big_name ${batch})"', Learning\ Rate = '"${ratio}"')}$'

    ./plot.py experiments/${ratio}-${batch}-{25,50,100}.txt -labels '25' '50' '100' \
      -folder charts/var1/hidden -title "${title}"
  done
done

echo 'ratio'
mkdir -p charts/var1/ratio

for batch in ${batch_sizes}; do
  for hidden in ${hidden_units}; do
    title=$'Learning Rate variation\n$_{('"$(big_name ${batch} latex)"', Hidden\ Units = '"${hidden}"')}$'

    ./plot.py experiments/{0.5,1.0,10.0}-${batch}-${hidden}.txt -labels '0.5' '1.0' '10.0' \
      -folder charts/var1/ratio -title "${title}"
  done
done

echo 'batch'
mkdir -p charts/var1/batch

for ratio in ${ratios}; do
  for hidden in ${hidden_units}; do
    title=$'Batch Size variation\n$_{(Learning\ Rate = '"${ratio}"', Hidden\ Units = '"${hidden}"')}$'

    ./plot.py experiments/${ratio}-{1,10,50,inf}-${hidden}.txt \
      -labels "$(short_name 1)" "$(short_name 10)" "$(short_name 50)" "$(short_name inf)" \
      -folder charts/var1/batch -title "${title}"
  done
done
