#!/bin/bash
# This script downloads the various pre-trained TensorFlow models that I'm
# interested in from the Model Detection Zoo.  While doing so, it also creates
# a "Model Size Proxy" MarkDown table located in ./models.

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
	echo "This script download TensorFlow models from the Object Detection Model"
  echo "Zoo to proj1/models/tensorflow, untars them, and removes the tar file."
  echo "While doing so, it records various proxy measures for model size, "
  echo "including (i) how long the download takes, (ii) how long the tar file"
  echo "takes to untar, (iii) the size of downloaded model directory, and "
  echo "finally (iv) the size of the model's frozen graph."
	echo
	echo "Usage: "
  echo "    * Deploy: $PROGNAME "
  echo "    * Test:   $PROGNAME --test "
	echo
	echo "Options:"
	echo
	echo "  -h, --help"
	echo "      This help text."
	echo
	echo "  --test"
	echo "      Run script in test mode."
	echo
	echo "  --"
	echo "      Do not interpret any more arguments as options."
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
	--test)
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
# Test or Deploy?
#------------------------------------------------------------------------------
if [ $test -eq 1 ]; then
  # Testing Environment 
  echo "This is a TEST.  Numeric values are fictional." 2>&1 | tee -a ${log}
  log="${logs}/tests/model_size_proxies.md"
  download_step='(time -p sleep "0.$(( $RANDOM % 100))")'
  untar_step='(time -p sleep "0.$(( $RANDOM % 100))")'
  dirsize_step='du -sh ${models} | grep -oE "[0-9\.]+[MG]"'
  graphsize_step=${dirsize_step}
else
  # Deployment Environment
  log="${logs}/model_size_proxies.md"
  download_step='(time -p wget -q ${the_zoo}/${animal} -P ${tf_models})'
  untar_step='(time -p tar -xf ${tarfile} -C ${tf_models} && rm ${tarfile})'
  dirsize_step='du -sh ${tf_models}/${animal%.tar.gz} | grep -oE "[0-9\.]+[MG]"'
  graphsize_step='du -sh ${tf_models}/${animal%.tar.gz}/frozen_inference_graph.pb'
fi

#------------------------------------------------------------------------------
# Log file
#------------------------------------------------------------------------------
if [ ! -f ${log} ]; then
  echo "CREATING LOG: ${log}"
  touch ${log}
else
  echo "APPENDING TO FILE: ${log}"
  echo -e "\n========================================================" 2>&1 |\
      tee -a ${log}
fi

#------------------------------------------------------------------------------
# Download, untar, and record model size proxies
#------------------------------------------------------------------------------
echo -e "`date`\n" 2>&1 | tee -a ${log}
echo "|  Model Name          |Wget Time|Tar Time|Dir Size|Graph Size|" 2>&1 | \
    tee -a ${log}
echo "|----------------------|---------|--------|--------|----------|" 2>&1 | \
    tee -a ${log}
for animal in ${zoo[@]}; do
  # Download file to models/tensorflow
  #   -- record download times in logfile 
  echo -n "| ${animal%.tar.gz} | " 2>&1 | tee -a ${log}
  eval $download_step 2>&1 | \
      grep real | cut -d" " -f2 | tr '\r\n' ' ' | tee -a ${log} 
  echo -n " |" 2>&1 | tee -a ${log}
  # Untar the file & remove tar.gz file
  #   -- record untarring times in logfile 
  tarfile=${tf_models}/${animal}
  eval $untar_step 2>&1 | \
      grep real | cut -d" " -f2 | tr '\r\n' ' ' | tee -a ${log}
  echo -n " |" 2>&1 | tee -a ${log}
  # Calculate size of model directory 
  #   -- record sizes in logfile 
  eval $dirsize_step 2>&1 | \
      tr '\r\n' ' ' | tee -a ${log}
  echo -n " |" 2>&1 | tee -a ${log}
  # Calculate size of frozen graph
  #   -- record sizes in logfile 
  eval $graphsize_step | \
      grep -oE "[0-9\.]+[MG]" 2>&1 | tr '\r\n' ' ' | tee -a ${log}
  echo " |" 2>&1 | tee -a ${log}
done
