#!/bin/bash
source ${BASH_SOURCE%/*}/../config.sh

# File name
readonly PROGNAME=$(basename $0)
# File name, without the extension
#readonly PROGBASENAME=${PROGNAME%.*}
# File directory
#readonly PROGDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Arguments
#readonly ARGS="$@"
# Arguments number
#readonly ARGNUM="$#"

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------
usage() {
	echo "This script converts TensorFlow frozen graphs to OpenVINO intermediate"
  echo "representations (IRs). Creates a MarkDown file with the output for each"
  echo "model and a summary table at top: Name, TF Size, IR Size, TF2IR Time."
	echo
	echo "Usage: "
  echo "  Deploy: "
  echo "    Run all models in project's zoo:  "
  echo "      $PROGNAME "
  echo "    Select a model "
  echo "      $PROGNAME -m <modelname>  "
  echo "  Test: " 
  echo "      $PROGNAME --test"
	echo "      $PROGNAME -m <modelname> --test "
	echo
	echo "Options:"
	echo
  echo "  -d, --data_type, -d=*, --data_type=*"
  echo "      Specify IR data type (FP16, FP32, etc)"
  echo "        * DEFAULT: FP16 (compatible with NCS2)"
	echo
  echo "  -g, --graph, -g=*, --graph=*"
  echo "      Specify graph file name. "
  echo "        * DEFAULT: frozen_inference_graph.pb"
  echo "        * NOTE: though not formally deprecated, this is not in use  "
	echo "          since all  models in my project zoo have frozen graphs.   "
  echo
	echo "  -h, --help"
	echo "      This help text."
	echo
	echo "  -l, --log, -l=*, --log=*"
  echo "      Override default log file name "
  echo '        * DEFAULT: tf_to_ir_conversions_${data_type}_<datetime>.md'
	echo
	echo "  -m, --model, -m=*, --model=*"
  echo '      Specify a single model name (without .tar.gz) from the Project'
  echo "      Zoo (my local subset of the Object Detection Model Zoo)."
  echo "        * DEFAULT: All models in Project Zoo (see below)."
	echo
	echo "  --test, --test-run, --dry-run"
	echo "      Run script in test mode."
	echo
	echo "  --"
	echo "      Do not interpret any more arguments as options."
	echo 
	echo "Project Zoo: "
  for animal in ${zoo[@]%.tar.gz}; do echo -e "\t$animal"; done
  echo
}

#------------------------------------------------------------------------------
# Parse Args
#------------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
	case "$1" in
	-h|--help)
		usage
		exit 0
		;;
  # Space-Delimited Arguments
	-d|--data_type)
    data_type="$2"
    shift
		;;
	-g|--graph)
		inference_graph="$2"
    shift
		;;
	-l|--log)
		logfile="$2"
    shift
		;;
	-m|--model)
		model="$2"
    shift
		;;
  # Equals-Delimited Arguments
	-d=*|--data_type=*)
		data_type="${1#*=}"
		;;
	-g=*|--graph=*)
		inference_graph="${1#*=}"
		;;
	-l=*|--log=*)
		logfile="${1#*=}"
		;;
	-m=*|--model=*)
		model="${1#*=}"
		;;
	--test|--test-run|--dry-run)
		test=1
    ;;
  --)
    break
    ;;
  # Do not interpret any more arguments as options.
  -*)
  		echo "Invalid option '$1'. Use --help to see the valid options" >&2
  		exit 1
  		;;
  # An option argument, continue
	*)
  esac
	shift
done

#------------------------------------------------------------------------------
# Derived Vars
#------------------------------------------------------------------------------
# DATA TYPE:
#     DEFAULT:  FP16
#     NOTE: Ultimate goal is to deploy on NCS2, so must be FP16
if [ -z $data_type ]; then
  data_type='FP16'
else
  # Ensure uppercase (fp->FP)
  data_type=`tr '[:lower:]' '[:upper:]' <<< "$data_type"` 
  # Ensure type is acceptable
  case $data_type in 
    FP16|FP32|INT8) ;; # DO NOTHING
    *) 
      echo '=================================================================='
      echo "ERROR: Bad data type: $data_type" 
      echo '=================================================================='
      echo
      usage; exit 0;; 
  esac
fi

# GRAPH: 
#     DEFAULT: Frozen Inference Graph
#     NOTE: No longer in use (all models in my zoo have frozen graph)
if [ -z $inference_graph ]; then
  inference_graph="frozen_inference_graph"
fi

# LOGFILE:
#     DEFAULT:  tf_to_ir_conversion_<DATATYPE>_<%Y%b%d-%Hh%Mm%Ss>.txt
#     OPTION:   Change name to whatever (suffix "%Y%b%d-%Hh%Mm%Ss" auto-added).
#     NOTE:     Log files are always put into ${proj1}/logs, so do not specify
#               path unless it is a valid subdir path of ${proj1}/logs
if [ -z ${logfile} ]; then
  logfile="tf_to_ir_conversions_${data_type}"
fi
timestamp=`date '+%Y%b%d-%Hh%Mm%Ss'`
logfile="${proj1}/logs/${logfile}_${timestamp}.md"
touch ${logfile}

# MODEL:
#     DEFAULT:  All models listed in my zoo.
#     OPTION:  Choose one model in zoo (e.g., if a run went bad, etc).
if [ ! -z ${model} ]; then
  # If the user has specific a model,
  #   then:  redefine zoo=model
  #   else:  use zoo sourced from config.sh
  unset zoo
  zoo=${model}      
fi

# TEST OR DEPLOY (PART 1):
#     DEFAULT:  Deploy
if [ -z ${test} ]; then test=0; fi
if [ ${test} -eq 1 ]; then
  echo "Data Type: ${data_type}"
  echo "Inference Graph: ${inference_graph}"
  echo "Log File: ${logfile}"
  echo "TF Ops Config File: ${config}"
  echo "Model Info:"
else
  intelopenvino
fi

#------------------------------------------------------------------------------
# CONVERSIONS
#------------------------------------------------------------------------------
for model in ${zoo[@]%.tar.gz}; do
  # Custom TF Operations Config File
  if [ ${model:0:3} == 'ssd' ]; then
    config=${ssd_v2_support}
  elif [ ${model:0:3} == 'fas' ]; then
    config=${faster_rcnn_support}
  else
    echo "Must choose an SSD or Faster RCNN model from Object Detection Zoo."
    exit 1
  fi
  tf_model=${tf_models}/${model}
  pipeline="${tf_model}/pipeline.config"

  # Test or Deploy: Part 2
  if [ ${test} -eq 1 ]; then
    echo -e "\tModel: ${model}"
    echo -e "\tInference Graph: ${inference_graph}"
    echo -e "\tTF Ops Config File: ${config}"
    echo -e "\tPath to TF Model: ${tf_model}"
    echo -e "\tTF Pipeline Config: ${pipeline}"
  else
    echo -e "\n\n-----------------------------------\n\n# ${model}\n\n"'```' 2>&1 | \
      tee -a ${logfile}
    python $mo/mo.py \
      --input_model "${tf_model}/${inference_graph}.pb" \
      --data_type "${data_type}" \
      --reverse_input_channels \
      --tensorflow_object_detection_api_pipeline_config ${pipeline} \
      --tensorflow_use_custom_operations_config $config \
      --output_dir ${ir_models} \
      --model_name ${model} 2>&1 | tee -a ${logfile}
    echo '```' 2>&1 | tee -a ${logfile}
  fi
done


#=========================================================================
# Create Table, Append to top of file, put output into 
if [ ${test} != 1 ]; then
  cat ${logfile} | grep "IR output name\|Total execution time" > tempfile1
  echo -e "${timestamp}\n\n" >> tempfile2
  echo "| Model Name | TF Size | IR Size | TF2IR Execution Time |" 2>&1| \
    tee -a tempfile2
  echo "|------------|---------|---------|----------------------|" 2>&1| \
    tee -a tempfile2
  while read line; do
    if [[ "$line" == *"IR output"* ]]; then
      model_name=`echo $line | rev | cut -d" " -f1 | rev`
      tf_size=`du -sh ${tf_models}/${model_name}/frozen_inference_graph.pb | grep -oE "[0-9\.]+[MG]"`
      ir_size=`du -sh ${ir_models}/${model_name}.bin | grep -oE "[0-9\.]+[MG]"`
    elif [[ "$line" == *"execution time"* ]]; then
      execution_time=`echo $line | grep -Eo "[0-9]+\.[0-9]*"`
      echo "| $model_name | $tf_size | $ir_size | $execution_time |" 2>&1 | \
        tee -a tempfile2
    else
      echo "|  **** SOMETHING | WENT | WRONG | HERE **** |" 2>&1 | \
        tee -a tempfile2
    fi
  done < tempfile1
  echo -e "\n\n\n" >> tempfile2
  cat tempfile2 | cat - ${logfile} > tempfile && mv tempfile ${logfile} 
  rm tempfile*
fi

