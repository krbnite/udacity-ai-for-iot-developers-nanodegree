#!/bin/bash

# This script simply maintains a list of environment variables referenced in
# some of the other bash scripts.

#------------------------------------------------------------------------------
# Project1 Absolute Paths
#------------------------------------------------------------------------------
#proj1="${HOME}/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter"
proj1=`git rev-parse --show-toplevel`/p01_people-counter
src="${proj1}/src"
logs="${proj1}/logs"
models="${proj1}/models"
tf_models="${models}/tensorflow"
ir_models="${models}/openvino"

#------------------------------------------------------------------------------
# Calling Directory (Abs Path)
#------------------------------------------------------------------------------
# Directory from which top-level script was called.
#   -- e.g., if in ${proj1}/models and call on someScript.sh from ${proj1}/src
#      (i.e., `bash ../src/someScript.sh`), then the calling directory will be
#      ${proj1}/models
abs_path_to_calling_dir=$PWD

#------------------------------------------------------------------------------
# Proj1 Paths Relative to Calling Directory
#------------------------------------------------------------------------------
# rel_path_from_script_to_config
#     Path of script that sourced config relative to the directory at the
#     top of teh call stack.
# rel_path_from_script_to_proj1_src:                          
#     Path of the directory that houses the script that sourced config relative 
#     to the directory at the top of teh call stack.
#
# Example
#     If you currently located in in ${proj1}/models (your "calling directory") and
#     call on someScript.sh from ${proj1}/src (i.e., `bash ../src/someScript.sh`), 
#     then the script directory's absolute path is ${proj1}/src, but its relative 
#     path is ../src/.
from_origin_to_script_to_config=${BASH_SOURCE}
from_origin_to_script_to_proj1=${BASH_SOURCE%/*}/..


#------------------------------------------------------------------------------
# Object Detection Zoo
#------------------------------------------------------------------------------
the_zoo="http://download.tensorflow.org/models/object_detection"
# Object Detection Zoo Tar Files
zoo=(\
  ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz \
  ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz \
  ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz \
  ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz \
  ssd_mobilenet_v2_coco_2018_03_29.tar.gz \
  ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz \
  ssd_inception_v2_coco_2018_01_28.tar.gz \
  faster_rcnn_inception_v2_coco_2018_01_28.tar.gz \
  faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz \
  faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz \
  faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz \
  faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz \
)



#------------------------------------------------------------------------------
# OpenVINO SetUp
#------------------------------------------------------------------------------
openvino='/opt/intel/openvino'
openvino_setup="${openvino}/bin/setupvars.sh"
function intelopenvino { source ${openvino_setup} -pyver 3.5; }
# must run intelopenvino to source environment
#
#------------------------------------------------------------------------------
# OpenVINO Model Optimizer
#------------------------------------------------------------------------------
mo="${openvino}/deployment_tools/model_optimizer"
mo_support="${mo}/extensions/front/tf"
ssd_v2_support="${mo_support}/ssd_v2_support.json"
faster_rcnn_support="${mo_support}/faster_rcnn_support.json"
mo_ssd_example='python ${mo}/mo.py --input_model ${inference_graph} --tensorflow_object_detection_api_pipeline_config ${pipeline} --reverse_input_channels --tensorflow_use_custom_operations_config ${ssd_v2_support} --output_dir ${ir_models} --model_name ${model}'
mo_frcnn_example='python ${mo}/mo.py --input_model ${inference_graph} --tensorflow_object_detection_api_pipeline_config ${pipeline} --reverse_input_channels --tensorflow_use_custom_operations_config ${faster_rcnn_support} --output_dir ${ir_models} --model_name ${model}'
#------------------------------------------------------------------------------
# OpenVINO: Run the IR Model
#------------------------------------------------------------------------------
# -- 3-server setup: Mosca, Web, FF
function mosca_server { node ${proj1}/webservice/server/node-server/server.js; }
function web_server { np run --prefix="${proj1}/webservice/ui" dev; }
function ffserver { ${proj1}/src/ext/ffmpeg/ffserver -f ./ffmpeg/server.conf; } # might need sudo
function sudo_ffserver { sudo ${proj1}/src/ext/ffmpeg/ffserver -f ./ffmpeg/server.conf; } 
# -- Model
cpu_ext_mac="${openvino}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
cpu_ext_linux="${openvino}/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
main_cpu_example='python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm;    # [1] In browser:  http://0.0.0.0:3004/'
main_ncs2_example='python3.5 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm;   # [1] In browser: http://0.0.0.0:3004/; [2] Note: NCS2 only accpets FP16 models.'
main_cam_example='python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm;    # [1] In browser: http://0.0.0.0:3004/;  [2] Must specify `-video_size`'




