ratios='0.5 1.0 10.0'
batch_sizes='1 10 50 inf'
hidden_units='25 50 100'

function contains {
  local word
  for word in "${@:2}"; do
    if [ "${word}" = "${1}" ]; then
      return word
    fi
  done
  return 1
}

function big_name_only {
  if [ "${1}" = "batch" ]; then
    if [ "${2}" = "1" ]; then
      if contains "latex" "${@}"; then
        echo "Stochastic\ Gradient\ Descent"
      else
        echo "Stochastic Gradient Descent"
      fi
    elif [ "${2}" = "inf" ]; then
      if [ "${3}" = "latex" ]; then
        echo "Gradient\ Descent"
      else
        echo "Gradient Descent"
      fi
    elif [[ "${2}" = "10" || "${2}" = "50" ]]; then
      if contains "latex" "${@}"; then
        echo "Mini\ Batch"
      else
        echo "Mini Batch"
      fi
    else
      echo "Gradient Type"
    fi
  elif [ "${1}" = "ratio" ]; then
    if contains "latex" "${@}"; then
      echo "Learning\ Rate"
    else
      echo "Learning Rate"
    fi
  elif [ "${1}" = "hidden" ]; then
    if contains "latex" "${@}"; then
      echo "Hidden\ Units"
    else
      echo "Hidden Units"
    fi
  fi

}

function big_name {
  if [[ "${1}" = "batch" && ( "${2}" = "1" || "${2}" = "inf" ) ]]; then
    echo "$(big_name_only "${@}")"
  else
    echo "$(big_name_only "${@}") = ${2}"
  fi
}

function short_name_only {
  if [ "${1}" = "batch" ]; then
    if [ "${2}" = "1" ]; then
      echo "SGD"
    elif [ "${2}" = "inf" ]; then
      echo "GD"
    elif [[ "${2}" = "10" || "${2}" = "50" ]]; then
      echo "MB"
    fi
  elif [ "${1}" = "ratio" ]; then
    echo "LR"
  elif [ "${1}" = "hidden" ]; then
    echo "HU"
  fi
}

function short_name {
  if [[ "${1}" = "batch" && ( "${2}" = "1" || "${2}" = "inf" ) ]]; then
    echo $(short_name_only "${@}")
  else
    echo "$(short_name_only "${@}") = ${2}"
  fi
}

folder=charts

if [ "${1}" != "" ]; then
  folder=${1}
  mkdir -p ${folder}
fi

echo var3
mkdir -p ${folder}/var3

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      labels=$(big_name batch ${batch})$'\n'$(big_name ratio ${ratio})$'\n'$(big_name hidden ${hidden})

      ./plot.py experiments/${ratio}-${batch}-${hidden}.txt \
        -labels "${labels}" -validate -folder ${folder}/var3 -title-size 10
    done
  done
done

echo var2

echo batch
mkdir -p ${folder}/var2/batch

for batch in ${batch_sizes}; do
  files=()
  labels=()

  for ratio in ${ratios}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ratio ${ratio}), $(short_name hidden ${hidden})")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/batch -title "$(big_name batch ${batch})" -legend-size 6
done

echo ratio
mkdir -p ${folder}/var2/ratio

for ratio in ${ratios}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for hidden in ${hidden_units}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name batch ${batch}), $(short_name hidden ${hidden})")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/ratio -title "$(big_name ratio ${ratio})" -legend-size 6
done

echo hidden
mkdir -p ${folder}/var2/hidden

for hidden in ${hidden_units}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for ratio in ${ratios}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name batch ${batch}), $(short_name ratio ${ratio})")
    done
  done

  ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/hidden -title "$(big_name hidden ${hidden})" -legend-size 6
done

echo var1

echo hidden
mkdir -p ${folder}/var1/hidden

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    title="$(big_name_only hidden)"$' variation\n$_{('"$(big_name batch ${batch} latex), $(big_name ratio ${ratio} latex))}$"

    ./plot.py experiments/${ratio}-${batch}-{25,50,100}.txt -labels '25' '50' '100' \
      -folder ${folder}/var1/hidden -title "${title}"
  done
done

echo ratio
mkdir -p ${folder}/var1/ratio

for batch in ${batch_sizes}; do
  for hidden in ${hidden_units}; do
    title="$(big_name_only ratio)"$' variation\n$_{('"$(big_name batch ${batch} latex), $(big_name hidden ${hidden} latex))}$"

    ./plot.py experiments/{0.5,1.0,10.0}-${batch}-${hidden}.txt -labels '0.5' '1.0' '10.0' \
      -folder ${folder}/var1/ratio -title "${title}"
  done
done

echo batch
mkdir -p ${folder}/var1/batch

for ratio in ${ratios}; do
  for hidden in ${hidden_units}; do
    title="$(big_name_only batch)"$' variation\n$_{('"$(big_name ratio ${ratio} latex), $(big_name hidden ${hidden} latex))}$"

    ./plot.py experiments/${ratio}-{1,10,50,inf}-${hidden}.txt \
      -labels "$(short_name batch 1)" "$(short_name batch 10)" "$(short_name batch 50)" "$(short_name batch inf)" \
      -folder ${folder}/var1/batch -title "${title}"
  done
done
