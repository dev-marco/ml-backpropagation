ratios='0.5 1.0 10.0'
batch_sizes='1 10 50 inf'
hidden_units='25 50 100'

lang="br"

function contains {
  local word
  for word in "${@:2}"; do
    if [ "${word}" = "${1}" ]; then
      return 0
    fi
  done
  return 1
}

function variation {
  if [ "${lang}" = "br" ]; then
    echo "Variação de ${@}"
  else
    echo "${@} variation"
  fi
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
        echo "Mini‑Batch"
      else
        echo "Mini-Batch"
      fi
    else
      if [ "${lang}" = "br" ]; then
        echo "Descida no Gradiente"
      else
        echo "Gradient Type"
      fi
    fi
  elif [ "${1}" = "ratio" ]; then
    if contains "latex" "${@}"; then
      if [ "${lang}" = "br" ]; then
        echo "Taxa\ de\ Aprendizado"
      else
        echo "Learning\ Rate"
      fi
    else
      if [ "${lang}" = "br" ]; then
        echo "Taxa de Aprendizado"
      else
        echo "Learning Rate"
      fi
    fi
  elif [ "${1}" = "hidden" ]; then
    if contains "latex" "${@}"; then
      if [ "${lang}" = "br" ]; then
        echo "Neurônios\ na\ Camada\ Oculta"
      else
        echo "Hidden\ Units"
      fi
    else
      if [ "${lang}" = "br" ]; then
        echo "Neurônios na Camada Oculta"
      else
        echo "Hidden Units"
      fi
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
    if [ "${lang}" = "br" ]; then
      echo "TA"
    else
      echo "LR"
    fi
  elif [ "${1}" = "hidden" ]; then
    if [ "${lang}" = "br" ]; then
      echo "CO"
    else
      echo "HU"
    fi
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

echo var1

echo ratio
mkdir -p ${folder}/var1/ratio

for batch in ${batch_sizes}; do
  for hidden in ${hidden_units}; do
    title="$(variation "$(big_name_only ratio)")"$'\n$_{('"$(big_name batch ${batch} latex), $(big_name hidden ${hidden} latex))}$"

    if ! ./plot.py experiments/{0.5,1.0,10.0}-${batch}-${hidden}.txt -labels '0.5' '1.0' '10.0' \
      -folder ${folder}/var1/ratio -title "${title}" -lang "${lang}"; then
      exit 1
    fi
  done
done

echo batch
mkdir -p ${folder}/var1/batch

for ratio in ${ratios}; do
  for hidden in ${hidden_units}; do
    title="$(variation "$(big_name_only batch)")"$'\n$_{('"$(big_name ratio ${ratio} latex), $(big_name hidden ${hidden} latex))}$"

    if ! ./plot.py experiments/${ratio}-{1,10,50,inf}-${hidden}.txt \
      -labels "$(short_name batch 1)" "$(short_name batch 10)" "$(short_name batch 50)" "$(short_name batch inf)" \
      -folder ${folder}/var1/batch -title "${title}" -lang "${lang}"; then
      exit 1
    fi
  done
done

echo hidden
mkdir -p ${folder}/var1/hidden

for batch in ${batch_sizes}; do
  for ratio in ${ratios}; do
    title="$(variation "$(big_name_only hidden)")"$'\n$_{('"$(big_name ratio ${ratio} latex), $(big_name batch ${batch} latex))}$"

    if ! ./plot.py experiments/${ratio}-${batch}-{25,50,100}.txt -labels '25' '50' '100' \
      -folder ${folder}/var1/hidden -title "${title}" -lang "${lang}"; then
      exit 1
    fi
  done
done

echo
echo var2

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

  if ! ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/ratio \
    -title "$(big_name ratio ${ratio})" -lang "${lang}"; then
    exit 1
  fi
done

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

  if ! ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/batch \
    -title "$(big_name batch ${batch})" -lang "${lang}"; then
    exit 1
  fi
done

echo hidden
mkdir -p ${folder}/var2/hidden

for hidden in ${hidden_units}; do
  files=()
  labels=()

  for batch in ${batch_sizes}; do
    for ratio in ${ratios}; do
      files+=("experiments/${ratio}-${batch}-${hidden}.txt")
      labels+=("$(short_name ratio ${ratio}), $(short_name batch ${batch})")
    done
  done

  if ! ./plot.py "${files[@]}" -labels "${labels[@]}" -folder ${folder}/var2/hidden \
    -title "$(big_name hidden ${hidden})" -lang "${lang}"; then
    exit 1
  fi
done

echo
echo var3
mkdir -p ${folder}/var3

for ratio in ${ratios}; do
  echo ${ratio}
  for batch in ${batch_sizes}; do
    echo ' '${batch}
    for hidden in ${hidden_units}; do
      echo '  '${hidden}
      labels=$(big_name ratio ${ratio})$'\n'$(big_name batch ${batch})$'\n'$(big_name hidden ${hidden})

      if ! ./plot.py experiments/${ratio}-${batch}-${hidden}.txt \
        -labels "${labels}" -validate -folder ${folder}/var3 \
        -title-size 10 -lang "${lang}"; then
        exit 1
      fi
    done
  done
done
