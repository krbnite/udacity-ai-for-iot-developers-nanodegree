#!/bin/bash
source ${BASH_SOURCE%/*}/../config.sh
echo
echo -e "Abs Path to Proj1:\n\t$proj1"; echo
echo -e "Abs Path to Proj1/src\n\t$src"; echo
echo -e "Abs Path to Proj1/models\n\t$models"; echo
echo -e "Abs Path to Proj1/models/tensorflow:\n\t$tf_models"; echo
echo -e "Abs Path to Proj1/models/openvino:\n\t$ir_models"; echo
echo -e "Abs Path to calling directory:\n\t$abs_path_to_calling_dir"; echo
echo -e "Rel Path From Calling Location to Script to Proj1"
echo -e      "\t$from_origin_to_script_to_proj1"; echo
echo -e "Rel Path From Calling Location to Script to Proj1/src/config:"
echo -e      "\t$from_origin_to_script_to_config"
echo
