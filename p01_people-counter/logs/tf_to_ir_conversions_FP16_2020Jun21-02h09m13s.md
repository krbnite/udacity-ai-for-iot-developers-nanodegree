2020Jun21-02h09m13s


| Model Name | TF Size | IR Size | TF2IR Execution Time |
|------------|---------|---------|----------------------|
| ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03 | 18M | 7.8M | 51.94 |
| ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03 | 10M | 6.6M | 48.92 |
| ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 | 49M | 59M | 77.51 |
| ssdlite_mobilenet_v2_coco_2018_05_09 | 19M | 8.5M | 66.36 |
| ssd_mobilenet_v2_coco_2018_03_29 | 66M | 33M | 97.33 |
| ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 | 128M | 113M | 75.60 |
| ssd_inception_v2_coco_2018_01_28 | 97M | 48M | 75.21 |
| faster_rcnn_inception_v2_coco_2018_01_28 | 55M | 26M | 158.27 |
| faster_rcnn_resnet50_lowproposals_coco_2018_01_28 | 115M | 56M | 217.08 |
| faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 | 238M | 128M | 359.68 |
| faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28 | 237M | 129M | 273.12 |
| faster_rcnn_nas_lowproposals_coco_2018_01_28 | 405M | 208M | 246.53 |






-----------------------------------

# ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.bin
[ SUCCESS ] Total execution time: 51.94 seconds. 
```


-----------------------------------

# ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.bin
[ SUCCESS ] Total execution time: 48.92 seconds. 
```


-----------------------------------

# ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.bin
[ SUCCESS ] Total execution time: 77.51 seconds. 
```


-----------------------------------

# ssdlite_mobilenet_v2_coco_2018_05_09

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssdlite_mobilenet_v2_coco_2018_05_09
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssdlite_mobilenet_v2_coco_2018_05_09.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssdlite_mobilenet_v2_coco_2018_05_09.bin
[ SUCCESS ] Total execution time: 66.36 seconds. 
```


-----------------------------------

# ssd_mobilenet_v2_coco_2018_03_29

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_mobilenet_v2_coco_2018_03_29
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v2_coco_2018_03_29.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_mobilenet_v2_coco_2018_03_29.bin
[ SUCCESS ] Total execution time: 97.33 seconds. 
```


-----------------------------------

# ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.bin
[ SUCCESS ] Total execution time: 75.60 seconds. 
```


-----------------------------------

# ssd_inception_v2_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	ssd_inception_v2_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_inception_v2_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_inception_v2_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_inception_v2_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 75.21 seconds. 
```


-----------------------------------

# faster_rcnn_inception_v2_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	faster_rcnn_inception_v2_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate Representation file is generated with the input image size of a fixed size.
Specify the "--input_shape" command line parameter to override the default shape which is equal to (600, 600).
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_v2_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_v2_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 158.27 seconds. 
```


-----------------------------------

# faster_rcnn_resnet50_lowproposals_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	faster_rcnn_resnet50_lowproposals_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_resnet50_lowproposals_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate Representation file is generated with the input image size of a fixed size.
Specify the "--input_shape" command line parameter to override the default shape which is equal to (600, 600).
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 217.08 seconds. 
```


-----------------------------------

# faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate Representation file is generated with the input image size of a fixed size.
Specify the "--input_shape" command line parameter to override the default shape which is equal to (600, 600).
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 359.68 seconds. 
```


-----------------------------------

# faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
[ WARNING ] Model Optimizer removes pre-processing block of the model which resizes image keeping aspect ratio. The Inference Engine does not support dynamic image size so the Intermediate Representation file is generated with the input image size of a fixed size.
Specify the "--input_shape" command line parameter to override the default shape which is equal to (600, 600).
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 273.12 seconds. 
```


-----------------------------------

# faster_rcnn_nas_lowproposals_coco_2018_01_28

```
Model Optimizer arguments:
Common parameters:
	- Path to the Input Model: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_nas_lowproposals_coco_2018_01_28/frozen_inference_graph.pb
	- Path for generated IR: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino
	- IR output name: 	faster_rcnn_nas_lowproposals_coco_2018_01_28
	- Log level: 	ERROR
	- Batch: 	Not specified, inherited from the model
	- Input layers: 	Not specified, inherited from the model
	- Output layers: 	Not specified, inherited from the model
	- Input shapes: 	Not specified, inherited from the model
	- Mean values: 	Not specified
	- Scale values: 	Not specified
	- Scale factor: 	Not specified
	- Precision of IR: 	FP16
	- Enable fusing: 	True
	- Enable grouped convolutions fusing: 	True
	- Move mean values to preprocess section: 	False
	- Reverse input channels: 	True
TensorFlow specific parameters:
	- Input model in text protobuf format: 	False
	- Path to model dump for TensorBoard: 	None
	- List of shared libraries with TensorFlow custom layers implementation: 	None
	- Update the configuration file with input/output node names: 	None
	- Use configuration file used to generate the model with Object Detection API: 	/Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/faster_rcnn_nas_lowproposals_coco_2018_01_28/pipeline.config
	- Operations to offload: 	None
	- Patterns to offload: 	None
	- Use the config file: 	/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
Model Optimizer version: 	2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.
The graph output nodes "num_detections", "detection_boxes", "detection_classes", "detection_scores" have been replaced with a single layer of type "Detection Output". Refer to IR catalogue in the documentation for information about this layer.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_nas_lowproposals_coco_2018_01_28.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/faster_rcnn_nas_lowproposals_coco_2018_01_28.bin
[ SUCCESS ] Total execution time: 246.53 seconds. 
```
