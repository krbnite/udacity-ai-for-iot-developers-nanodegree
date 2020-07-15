#!/bin/bash

# This script downloads the various pre-trained TensorFlow models that I'm
# interested in from the Model Detection Zoo.  While doing so, it also creates
# a "Model Size Proxy" MarkDown table located in ./models.

source ${BASH_SOURCE%/*}/../config.sh

log="${logs}/tests/model_size_proxies.md"
if [ ! -f ${log} ]; then
    echo "CREATING LOG: ${log}"
    touch ${log}
else
    echo "APPENDING TO FILE: ${log}"
    echo -e "\n========================================================" 2>&1 |\
        tee -a ${log}
fi

echo "This is a TEST.  Numeric values are fictional." 2>&1 | tee -a ${log}
echo -e "`date`\n" 2>&1 | tee -a ${log}
echo "|  Model Name             |  Wget Time | Tar Time | Dir Size |" 2>&1 |\
    tee -a ${log}
echo "|-------------------------|------------|----------|----------|" 2>&1 |\
    tee -a ${log}
for animal in ${zoo[@]}; do
    # Download file to models/tensorflow
    #   -- record download times in logfile 
    echo -n "| ${animal%.tar.gz} | " 2>&1 | tee -a ${log}
    (time -p sleep "0.$(( $RANDOM % 100))") 2>&1 |\
        grep real | cut -d" " -f2 | tr '\r\n' ' ' | tee -a ${log} 
    echo -n " |" 2>&1 | tee -a ${log}
    # Untar the file & remove tar.gz file
    #   -- record untarring times in logfile 
    tarfile=${tf_models}/${animal}
    (time -p sleep "0.$(( $RANDOM % 100))") 2>&1 |\
        grep real | cut -d" " -f2 | tr '\r\n' ' ' | tee -a ${log} 
    echo -n " |" 2>&1 | tee -a ${log}
    # Calculate size of model directory 
    #   -- record sizes in logfile 
    du -sh ${models} | grep -oE "[0-9\.]+[MG]" 2>&1 |\
        tr '\r\n' ' ' | tee -a ${log}
    echo -n " |" 2>&1 | tee -a ${log}
    # Calculate size of frozen graph
    #   -- record sizes in logfile 
    du -sh ${models} | grep -oE "[0-9\.]+[MG]" 2>&1 |\
        tr '\r\n' ' ' | tee -a ${log}
    echo " |" 2>&1 | tee -a ${log}
done
