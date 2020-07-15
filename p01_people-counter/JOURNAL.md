# Some Specs
On my MacBook:
* Node: v12.3.1 
* NPM: 6.9.0
* Conda Environment:  Python3.5 
  - Create env  
    ```
    conda create -n py35 python=3.5      # What Udacity used
    conda activate py35
    conda install numpy                  # pip3
    conda install -c sci-bots paho-mqtt  # pip3
    conda install cmake                  # brew
    conda install zeromq                 # brew
    ```
* FFMPEG
  - followed directions
  - on `./configure` step, I got error ("nasm/yasm not found or too old. Use 
    `--disable-x86asm` for a crippled build."), which I googled and found that
    the solution was to install `yasm`, which I did like `conda install yasm`
  - ultimately, installation looked like this:
    ```
    mkdir bin
    git clone https://git.ffmpeg.org/ffmpeg.git bin/ffmpeg
    cd bin/ffmpeg
    git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
    ./configure
    # -- received ERROR
    conda install yasm
    make -j4
    ```
  - LOTS of deprecations warnings...but moving on!
* NPM
  - There are three components that need to be running in separate terminals 
    for this application to work:
    * MQTT Mosca server
    * Node.js\* Web server
    * FFmpeg server
  - From the main directory:
    * For MQTT/Mosca server:
      ```
      cd webservice/server
      npm install
      ```
    * For Web server:
      ```
      cd ../ui
      npm install
      ```

    * Note: If any configuration errors occur in mosca server or Web server 
      while using npm install, use the below commands:
      ```
      sudo npm install npm -g
      rm -rf node_modules
      npm cache clean
      npm config set registry "http://registry.npmjs.org"
      npm install
      ```

Additional Python Libraries
```
pip install --upgrade setuptools
pip install cython
pip install tensorflow
conda install jupyter
conda install networkx
```

NOTES
* had to upgrade setuptools and install cython because the first time I tried 
  to pip install tensorflow, I got error messages telling me I had to
  - pip installed cython b/c anything to do w/ TensorFlow seems to work
    better if pip installed
* had to install `jupyter` to use ipython shells when
  in the py35 environment
* had to install `networkx` after running the model optimizer
  and choking on an error
* trying to get TF2.2 to work with everything was a disaster; e.g., numpy
  had to be version 1.16.0 or better, but that wasn't supported on py35; had 
  to pip uninstall...then I conda installed to see what would happen.... Conda
  went with giving me TF1.10...which sucks, but I guess that's what's available 
  for py35 (even though the website says py35 is fine: https://www.tensorflow.org/install/pip).
  


------------------------------------

# Models
Since I almost exclusively use TensorFlow/Keras, I am plucking models from 
the TensorFlow Model Zoo.  


## TensorFlow Hub
TF Hub:  I initially thought to check out the TensorFlow Hub,
but it has a more limited selection of object detection models, and its focus
is on frictionlessly delivering the model into your TF environment. 

Promising candidates (if it comes down to it):
* https://tfhub.dev/google/faster\_rcnn/openimages\_v4/inception\_resnet\_v2/1
* https://tfhub.dev/google/openimages\_v4/ssd/mobilenet\_v2/1
* https://tfhub.dev/google/object\_detection/mobile\_object\_localizer\_v1/1

## TF Model Zoo  
The Model Zoo has a fairly large collection of models. Importantly, 
downloading a model from the Model Zoo directly and explicitly provides the things 
I need convert the model to an OpenVINO IR.

After looking through the main page a bit, I looked through some of the 
official models, as well as the research models.  In the research collection,
there is an overwhelming amount of models.  Fortunately, there is a subdirectory
called `object_detection` that focuses the search.  Another subdirectory,
`deeplab`, also appears to have promising model candidates, however, these are 
semantic segmentation models, so they might be fairly slow and resource
demanding (also, on a quick google search, I saw some evidence that all person
pixels blend into each other, which would reduce the counting accuracy; while
I'm sure the model can be re-purposed to delineate between people, I don't 
really have a lot of time to figure it out).



## The Detection Model Zoo 

Inside the un-tar'ed directory:
* a graph proto (graph.pbtxt)
* a checkpoint (model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta)
* a frozen graph proto with weights baked into the graph as constants 
  (frozen\_inference\_graph.pb) to be used for out of the box inference (try this 
  out in the Jupyter notebook!)
* a config file (pipeline.config) which was used to generate the graph. These 
  directly correspond to a config file in the samples/configs) directory but 
  often with a modified score threshold. In the case of the heavier Faster R-CNN 
  models, we also provide a version of the model that uses a highly reduced number 
  of proposals for speed.

### Choosing a Model
There are many models to choose from.  Without looking carefully at each one, it's
tough to really know what will be best.  However, the Detection Model Zoo provides
some tables specifying speed and accuracy (the mAP score) metrics, which are
defined as follows:

* speed:   time in ms it takes to perform inference on a 600x600 image
* accuracy:  the mean average precision (mAP)
  - the average precision is the average of the precision values for given recall 
    values ranging between 0 to 1 (i.e., this is similar to an integration over 
    the precision-recall curve)
  - the mean average precision is the average of the average precisions over 
    multiple classes

Given these metrics, I can narrow down the potential choices by coming up with
some sensible restrictions.

####  Attempt #1
I started this process out by defining two restrictions:
* R1:  only models that clocked under 100ms per 600x600 image
* R2:  only models w/ mAP => 20


These restrictions immediately ruled out:
* ssd\_mobilenet\_v1\_0.75\_depth\_coco
* ssd\_mobilenet\_v1\_quantized\_coco
* ssd\_mobilenet\_v1\_0.75\_depth\_quantized\_coco
* faster\_rcnn\_resnet101\_coco
* faster\_rcnn\_inception\_resnet\_v2\_atrous\_coco
* faster\_rcnn\_inception\_resnet\_v2\_atrous\_lowproposals\_coco
* faster\_rcnn\_nas
* faster\_rcnn\_nas\_lowproposals\_coco
* mask\_rcnn\_inception\_resnet\_v2\_atrous\_coco
* mask\_rcnn\_resnet101\_atrous\_coco
* mask\_rcnn\_resnet50\_atrous\_coco


To further rule out models before beginning, I tightened the
restrictions:
* R1:  only models that clocked \<= 50ms per 600x600 image
* R2:  only models w/ mAP \> 25


Models kicked out for low speed:
* ssd\_mobilenet\_v1\_fpn\_coco
* ssd\_resnet\_50\_fpn\_coco
* faster\_rcnn\_inception\_v2\_coco
* faster\_rcnn\_resnet50\_coco
* faster\_rcnn\_resnet50\_lowproposals\_coco
* rfcn\_resnet101\_coco
* faster\_rcnn\_resnet101\_lowproposals\_coco
* mask\_rcnn\_inception\_v2\_coco

Remaining models kicked out for low mAP:
* ssd\_mobilenet\_v1\_coco
* ssd\_mobilenet\_v1\_ppn\_coco
* ssd\_mobilenet\_v2\_coco
* ssd\_mobilenet\_v2\_quantized\_coco
* ssdlite\_mobilenet\_v2\_coco
* ssd\_inception\_v2\_coco

This left me with 0 models left to choose from.

#### Attempt #2
There is clearly an inverse relationship between speed and mAP, so it
might be best to decide on an intermediate speed:

* RESTRICTION:  look at models where `30 < speed (ms) < 100`

This restriction slimmed down the pile pretty efficiently, immediately
rejecting the following models:
* ssd\_mobilenet\_v1\_0.75\_depth\_coco
* ssd\_mobilenet\_v1\_quantized\_coco
* ssd\_mobilenet\_v1\_0.75\_depth\_quantized\_coco
* ssd\_mobilenet\_v1\_ppn\_coco
* ssd\_mobilenet\_v2\_quantized\_coco
* ssdlite\_mobilenet\_v2\_coco
* faster\_rcnn\_resnet101\_coco
* faster\_rcnn\_inception\_resnet\_v2\_atrous\_coco
* faster\_rcnn\_inception\_resnet\_v2\_atrous\_lowproposals\_coco
* faster\_rcnn\_nas
* faster\_rcnn\_nas\_lowproposals\_coco
* mask\_rcnn\_inception\_resnet\_v2\_atrous\_coco
* mask\_rcnn\_resnet101\_atrous\_coco
* mask\_rcnn\_resnet50\_atrous\_coco

The remaining models (ordered by `Speed - mAP`):


Model Name                                                  | Speed (ms) | COCO mAP[^1] | Speed - mAP | Outputs
----------------------------------------------------------- | :--------: | :----------: | :---------: | :---:
[ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)                                                                       | 30         | 21           |  9  | Boxes
[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)                                                                       | 31         | 22           | 9  | Boxes
[ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)                                                                       | 42         | 24           | 18  | Boxes
[ssd_mobilenet_v1_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                         | 56         | 32           | 24  | Boxes
[faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                       | 58         | 28           | 30  | Boxes
[ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                             | 76         | 35           | 41  | Boxes
[mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                           | 79         | 25           |  54 | Masks
[faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)                                                               | 89         | 30           | 59  | Boxes
[rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)                                                                           | 92         | 30           | 62  | Boxes
[faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)                                     | 64         |              |  ? | Boxes
[faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz)                                   | 82         |              |  ? | Boxes

 

The highest mAP score left:
* ssd\_resnet\_50\_fpn\_coco w/ (speed, mAP) = (76, 35) 

The fastest speed is technically ssd\_mobilenet\_v1\_coco, but for only
1ms in cost, we get an extra mAP point from `v2` (both models have the
same `speed-mAP` score, which is the lowest of the group):
* ssd\_mobilenet\_v2\_coco w/ (speed, mAP) = (31,22)

Though the Faster RCNN `lowproposals` models do not have a mAP score listed, the corresponding
models (with higher proposals) score between 30-32 mAP, so it seems ok to assume that
the low-proposal models do not do much better (and maybe even worse).

Honestly, I wanted to see if the quantized or reduced depth models worked well. Fortunately,
there is one that has both:
* [ssd_mobilenet_v1_0.75_depth_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz)
  - (speed, mAP) = (29, 16)


### Final Model List (Preliminary)
I will scope the following models out.

Model Name                                                  | Speed (ms) | COCO mAP[^1] | Speed - mAP | Outputs
----------------------------------------------------------- | :--------: | :----------: | :---------: | :---:
[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)                                                                       | 31         | 22           | 9  | Boxes
[ssd_mobilenet_v1_0.75_depth_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz)            | 29         |  16          | 13 | Boxes
[ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                             | 76         | 35           | 41  | Boxes


### TensorFlow Model Zoo (References)

* Main Page:  https://github.com/tensorflow/models
* Official Models:  https://github.com/tensorflow/models/tree/master/official
* Research Models:  https://github.com/tensorflow/models/tree/master/research
  - Object Detection:  https://github.com/tensorflow/models/tree/master/research/object_detection
    * Documentation:  https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc
    * Detection Model Zoo:  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - DeepLab: https://github.com/tensorflow/models/tree/master/research/deeplab
    * Documentation:  https://github.com/tensorflow/models/tree/master/research/deeplab/g3doc
    * DeepLab Model Zoo:  https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md


## Download and Inspect the Models

```
mkdir -p models/tensorflow
cd models/tensorflow
zoo="http://download.tensorflow.org/models/object_detection"
# SSD MobileNet V2 (COCO)
time wget $zoo/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# SSD MobileNet V1 (0.75 depth, quantized, COCO)
time wget $zoo/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz
# SSD ResNet50 FPN (COCO)
time wget $zoo/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
```

The `time` command measures how long each model took to download:
| Model | Time to Download (sec) |
|-------|------------------------|
| SSD MobileNet V2       | 21.1  |
| SSD MobileNet V1 RD&Q  | 5.5   |
| SSD ResNet50 FPN       | 51.5  |


Clearly, the RD&Q model must be much smaller than the others, while the ResNet50
model is much bigger.  Given just these numbers, I might wager that the SSD.MN.V2 
model is going to be a good compromise between storage, speed, and accuracy.

Let's get some more numbers from untarring these files.

```
# Untar
time tar -xvf ssd_mobilenet_v2*tar.gz
time tar -xvf ssd_mobilenet_v1*tar.gz
time tar -xvf ssd_resnet50*tar.gz
# Clean up *tar.gz files
rm *tar.gz
```

| Model | Time to Untar (sec) |
|-------|---------------------|
| SSD MobileNet V2      | 1.1 |
| SSD MobileNet V1 RD&Q | 0.5 |
| SSD ResNet50 FPN      | 2.2 |

Again, the untarring process shows that the RD&Q model is the most lightweight, while
the ResNet50 model is our heavy weight.

Finally, we can look at the model sizes directly.

```
du -sh ssd_*
    79M    ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
   201M    ssd_mobilenet_v2_coco_2018_03_29
   386M    ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
```

There is one caveat: the RD&Q model has no frozen graph:
```
find $PWD -name frozen* | rev | cut -d/ -f1-2 | rev
    ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb
    ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
```

Looking closer, I found that it only has TFLite models:

```
ls ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18/
   model.ckpt.data-00000-of-00001 model.ckpt.meta                tflite_graph.pb
   model.ckpt.index               pipeline.config                tflite_graph.pbtxt
```

It *might* work to just run the Model Optimizer on `tflite_graph.pb`, however I've
googled this a bit (e.g., "how to convert tflite model to openvino intermediate 
representation") and haven't really found much on it.  

So it look like we might have a face off between:
* SSD MobileNet V2
* SSD ResNet50 V1 FPN


-------------------------------------------------------------------------------



# The Model Optimizer

TensorFlow models have some very specific things that need to be accounted
for when converting to IR using OpenVINO's Model Optimizer, which I wrote
about during phase 1 and which you can read all about on the OpenVINO website:
* Krbnite: [The Model Optimizer (Intel at the Edge)](https://krbnite.github.io/Intel-at-the-Edge-The-Model-Optimizer/)
* OpenVINO: [Converting TF Object Detection Models](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

The gist is that frozen TF models are easiest to deal with. For unfrozen graphs, 
you have to specify all kinds of things (means, scales, etc). For a frozen
graph from the TF Detection Model Zoo, you will likely need the following 
flags:
* `--tensorflow_use_custom_operations_config` 
  - this flag takes the argument `<path_to_subgraph_replacement_configuration_file.json>`,
    which is a subgraph replacement config file that describes rules for converting
    the TF model to the IR
  - the required JSON file should be housed within a subdirectory of the model
    optimizer's directory in the OpenVINO installation
  - specifically, for the models from TensorFlow's Object Detection API zoo, the
    config files can be found in:
    `/deployment_tools/model_optimizer/extensions/front/tf directory`
* `--tensorflow_object_detection_api_pipeline_config` 
* for most (if not all) TF models, you'll have to flag `--reverse_input_channels`

For more information:
* OpenVINO: [Converting TF Object Detection Models](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)


Let's check out the config files!  


```
ls /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf | grep ssd
    ssd_support.json
    ssd_support_api_v1.14.json
    ssd_toolbox_detection_output.json
    ssd_toolbox_multihead_detection_output.json
    ssd_v2_support.json
```

The OpenVINO docs page mentioned above recommends to use `ssd_v2_support.json` for frozen 
SSD topologies from the models zoo.  Technically all 3 of the models I chose are SSD 
models, though I suspect only two of them will convert properly.  (No harm in trying all
3 though.)


But first, we need to setup OpenVINO.



-------------------------------------------------------------------------------


# Start OpenVINO

To use OpenVINO, you must source it:
```
source /opt/intel/openvino/bin/setupvars.sh
```

For convenience, I have the following in my `.bash_profile`:
```
function intelopenvino {
  source /opt/intel/openvino/bin/setupvars.sh
}
echo "To use openvino: intelopenvino"  # default to on for now...
```

Then if/when I want to use it:
```
intelopenvino
```

For this project, it is suggested to use python 3.5, which I
instantiated a conda environment with (`py35`).  

```
conda activate py35
```

-------------------------------------------------------------------------------


# Attempted Model Optimizations: Debugging the Environment

First, we need to setup some convenient environment variables:

```
# Path to Project1 Models
models="${HOME}/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models"

# Path to Models
ssd_mn_v2="ssd_mobilenet_v2_coco_2018_03_29"
ssd_mn_v1_rq="ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18"
ssd_rn50_v1_fpn="ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"

# Path to Model Optimizer
mo="/opt/intel/openvino/deployment_tools/model_optimizer"

# Path to Config File
config="${mo}/extensions/front/tf/ssd_v2_support.json"
```

Now let's try to convert these models one at time.  We'll start with SSD 
MobileNet V2.

```
# SSD MobileNet V2
#=================================================================
# Inputs
#=================================================================
model=${ssd_mn_v2}
inference_graph=frozen_inference_graph.pb
#=================================================================
tf_model=${models}/tensorflow/${model}
inference_graph="${tf_model}/${inference_graph}"
pipeline="${tf_model}/pipeline.config"
python $mo/mo.py \
  --input_model ${inference_graph} \
  --reverse_input_channels \
  --tensorflow_object_detection_api_pipeline_config ${pipeline} \
  --tensorflow_use_custom_operations_config $config \
  --output_dir ${models}/openvino/ \
  --model_name ${model}
```


This first run triggered a conda environment nightmare...

Here is the output (errors at the bottom).

```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
        - Path for generated IR:        $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/.
        - IR output name:       frozen_inference_graph
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       True
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
        - Operations to offload:        None
        - Patterns to offload:  None
        - Use the config file:  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version:        2019.3.0-408-gac8584cb7
[ ERROR ]  Error happened while importing tensorflow module. It may happen due to unsatisfied requirements of that module. Please run requirements installation script once more.
Details on module importing failure: numpy.ufunc has the wrong size, try recompiling. Expected 192, got 216
[ ERROR ]  
Detected not satisfied dependencies:
        tensorflow: package error, required: 1.2.0
        tensorflow: not installed, required: 2.0.0
        networkx: installed: 2.4, required: 2.4

Please install required versions of components or use install_prerequisites script
/opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_tf.sh
Note that install_prerequisites scripts may install additional components.
```

I actually didn't read that last part about the `install_prerequisites` 
script.  However, though I had previously installed TensorFlow, it didn't 
install correctly (e.g., when importing it into python, it would throw
a bunch of errors).  So I pip uninstalled, the pip installed again.  Note
that I was pip installing the latest, TF2.2.  Anyway, same errors.  It came 
down to the numpy version in my environment being too old.  When I tried to
upgrade numpy, I couldn't.  I found that python 3.5 could not support the 
version of numpy required by TF2.2.  Figured that I didn't need the latest,
greatest TF: I'll see what conda thinks I need.  So I pip uninstalled 
tensorflow, then conda installed it.  Conda decided I needed TF1.10, which
is a pretty dang old version of TF.  However, it worked: (i) no more TF-related
errors when trying to run the model optimizer, and (ii) no more errors when
manually importing tensorflow into an ipython session. Finally, I conda
installed networkx, then attempt to run the model optimizer again.

One more snag: `networkx: installed: 2.4, required: 2.4`

What a weird snag, right?  I remember a lot of people in phase 1 having this
issue...  Anyway, this was resolved by conda uninstaling network, then
reinstalling an older version: `conda install networkx=2.3`.  It worked.  (Got
the advice here: https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/831696)

-------------------------------------------------------------------------------


# Model Optimization

## SSD MobileNet V2

```
# SSD MobileNet V2
#=================================================================
# Inputs
#=================================================================
model=${ssd_mn_v2}
inference_graph=frozen_inference_graph.pb
#=================================================================
tf_model=${models}/tensorflow/${model}
inference_graph="${tf_model}/${inference_graph}"
pipeline="${tf_model}/pipeline.config"
python $mo/mo.py \
  --input_model ${inference_graph} \
  --reverse_input_channels \
  --tensorflow_object_detection_api_pipeline_config ${pipeline} \
  --tensorflow_use_custom_operations_config $config \
  --output_dir ${models}/openvino/ \
  --model_name ${model}
```

Output:
```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb
        - Path for generated IR:        $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/.
        - IR output name:       frozen_inference_graph
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       True
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
        - Operations to offload:        None
        - Patterns to offload:  None
        - Use the config file:  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version:        2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: $HOME/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 63.42 seconds. 
```



## SSD ResNet50 V1 FNP

```
# SSD ResNet50 V1 FNP
#=================================================================
# Inputs
#=================================================================
model=${ssd_rn50_v1_fpn}
inference_graph=frozen_inference_graph.pb
#=================================================================
tf_model=${models}/tensorflow/${model}
inference_graph="${tf_model}/${inference_graph}"
pipeline="${tf_model}/pipeline.config"
python $mo/mo.py \
  --input_model ${inference_graph} \
  --reverse_input_channels \
  --tensorflow_object_detection_api_pipeline_config ${pipeline} \
  --tensorflow_use_custom_operations_config $config \
  --output_dir ${models}/openvino/ \
  --model_name ${model}
```

Output:
```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb
        - Path for generated IR:        /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/
        - IR output name:       ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       True
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/pipeline.config
        - Operations to offload:        None
        - Patterns to offload:  None
        - Use the config file:  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version:        2019.3.0-408-gac8584cb7
The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.xml
[ SUCCESS ] BIN file: /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.bin
[ SUCCESS ] Total execution time: 73.61 seconds. 
```



## SSD MobileNet V1 (Reduced Depth & Quantized)
This is the one I suspect will not work.

```
# SSD MobileNet V1 R&Q
#=================================================================
# Inputs
#=================================================================
model=${ssd_mn_v1_rq}
inference_graph=tflite_graph.pb
#=================================================================
tf_model=${models}/tensorflow/${model}
inference_graph="${tf_model}/${inference_graph}"
pipeline="${tf_model}/pipeline.config"
python $mo/mo.py \
  --input_model ${inference_graph} \
  --reverse_input_channels \
  --tensorflow_object_detection_api_pipeline_config ${pipeline} \
  --tensorflow_use_custom_operations_config $config \
  --output_dir ${models}/openvino/ \
  --model_name ${model}
```

As expected, this did not work (however, there is a twist -- read on).

Output (ERRORS):
```
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18/tflite_graph.pb
        - Path for generated IR:        /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/openvino/
        - IR output name:       ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:       False
        - Reverse input channels:       True
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  /Users/kevinurban/GitHub/udacity-ai-for-iot-developers-nanodegree/p01_people-counter/models/tensorflow/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18/pipeline.config
        - Operations to offload:        None
        - Patterns to offload:  None
        - Use the config file:  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Model Optimizer version:        2019.3.0-408-gac8584cb7
[ ERROR ]  Failed to match nodes from custom replacement description with id 'ObjectDetectionAPIPreprocessorReplacement':
It means model and custom replacement description are incompatible.
Try to correct custom replacement description according to documentation with respect to model node names
[ ERROR ]  Failed to match nodes from custom replacement description with id 'ObjectDetectionAPISSDPostprocessorReplacement':
It means model and custom replacement description are incompatible.
Try to correct custom replacement description according to documentation with respect to model node names
[ ERROR ]  Cannot infer shapes or values for node "TFLite_Detection_PostProcess".
[ ERROR ]  Op type not registered 'TFLite_Detection_PostProcess' in binary running on Kevins-MBP.home. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.
[ ERROR ]  
[ ERROR ]  It can happen due to bug in custom shape infer function <function tf_native_tf_node_infer at 0x13c55f0d0>.
[ ERROR ]  Or because the node inputs have incorrect values/shapes.
[ ERROR ]  Or because input shapes are incorrect (embedded to the model or passed via --input_shape).
[ ERROR ]  Run Model Optimizer with --log_level=DEBUG for more information.
[ ERROR ]  Op type not registered 'TFLite_Detection_PostProcess' in binary running on Kevins-MBP.home. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.
Stopped shape/value propagation at "TFLite_Detection_PostProcess" node. 
 For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #38. 
Exception occurred during running replacer "REPLACEMENT_ID" (<class 'extensions.middle.PartialInfer.PartialInfer'>): Op type not registered 'TFLite_Detection_PostProcess' in binary running on Kevins-MBP.home. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed.
Stopped shape/value propagation at "TFLite_Detection_PostProcess" node. 
 For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #38. 
```

The twist: yes, this model is not supported, however there is a similar reduced
model that is supported, as well as multiple quantized TFLite models (however, it
appears that all the quantized models are for image classification, not object
detection).

This OpenVINO docs page on 
[Converting a TensorFlow Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)
lists the following 
* [SSD MobileNet V1 0.75 Depth COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz)

There are also the following models from my list above that would be interesting 
to compare, as well as some I rejected that would be interesting for comparison
as well.   The table below includes these, as well as the aforementioned reduced 
depth SSD MobileNet model and the two models for which I've already created IRs.
However, one thing I found is that not all models in the Model Detection Zoo
are supported -- not just the reduced depth, quantized model I tried above.  
Basically, you have to cross-refence the Model Detection Zoo models with the
OpenVINO docs page, [Converting a TensorFlow Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html). (Or, you can just use that OpenVINO
page...since it contains everything in the intersection and has them all listed 
under "Supported Frozen Topologies from TensorFlow Object Detection Models Zoo."
That said, it doesn't compare and contrast speed and accuracy metrics...so both
pages are useful.  

So, for example, none of the quantized models from the Model Detection Zoo are
supported (the only quantized models that are supported are TFLite ones listed
elsewhere).


| Model Name | Speed (ms) | mAP |
|------------|------------|-----|
| **SSD MobileNet V1** | - | - |
| [SSD MobileNet V1 0.75 Depth COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz)                                  | 26         | 18 |
| [ssd_mobilenet_v1_ppn_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)                           | 26         | 20 |
| [ssd_mobilenet_v1_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                         | 56         | 32 |
| **SSD MobileNet V2** | - | - |
| [ssdlite_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)                                                               | 27         | 22 |
| [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)                                                                       | 31         | 22 | 
| **SSD ResNet50 V1** | - | - |
| [ssd_resnet_50_fpn_coco ☆](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)                             | 76         | 35 | 
| **SSD Inception V2** | - | - |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)                                                                       | 42         | 24 |
| **Faster RCNN Inception V2** | - | - |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                                       | 58         | 28 |
| **Faster RCNN ResNet (Low Proposals)** | - | - |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)                                     | 64         | ?  | 
| **Faster RCNN Inception ResNet (Regular vs Low Proposals)** | - | - |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)                           | 620        | 37 |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | 241        | ?  |
| **Faster RCNN NAS (Low Proposals)** | - | - |
| [faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz)                                               | 540        | ?  |

The reduced depth model has the best time (but not much better than the SSD 
MobileNet V2 model I already have), but -- dang! -- its mAP is so low.

But I like the idea of comparing models in general once I get this app to be
plug-and-play...so I'll probably keep it.  Will probably keep all the SSD models
to see what models best pair with it: MobileNet, ResNet, Inception? Then
there are the Faster RCNN models, which pair with Inception and ResNet, so
we can invert the question, e.g., which model pairs best with Inception:
SSD or Faster RCNN?

The Faster RCNN models generally have slightly 
higher mAP scores, but typically have much longer inference times.  This is
especially true for the Faster RCNN NAS model, which has the highest mAP score
of 43, but also the longest inference time by far: 1833 ms (i.e., 1.833 seconds).
However, the F-RCNN-NAS Low Proposals model cuts the inference time by 70% 
(down to 540 ms, i.e., about 0.5 seconds).  So, just for comparative purposes,
this model is worth looking at.  

I was going to drop the Faster RCNN Inception ResNet model and just keep its
LP version, but its inference time isn't much worse that the LP Faster RCNN NAS 
model (620ms vs 540ms), and it shows a relatively high mAP score (37). So, it's
worth taking a look at.  That said, the LP version of Faster RCNN Inception
ResNet cuts the inference time down to 241ms, so worth looking at that too 
(wish mAP scores were included on these LP models). 

I am leaving out Faster RCNN ResNet50 since SSD ResNet50 FPN does so much
better (35 vs 30) at a slightly faster rate (76ms vs 89ms). Similarly,
the Faster RCNN ResNet101 model seems to do relatively poorly as well:
the extra parameters bump its inference time to 106ms, while barely improving 
the mAP score (32). Both Faster RCNN ResNet models have "low proposal" versions,
which cut down the inference time quite a bit (24-25ms each). Considering this,
I'm going to look at the low proprosals version of Faster RCNN ResNet50, which 
quickens the model's speed to 64ms.  There is no provided mAP score for this
one, and I suspect it could perform worse than its high proposal cousin...so
we will see. (No need to do the Faster RCNN ResNet101 Low Proposals model, 
IMHO.)

As a point of sanity, I'm not even going to look at the "mask output" models,
and just stick with the "box output" models above. As a second point of 
sanity, I won't bother looking at these models pre-trained on the other
datasets (e.g., kitti, open images, ava).


# Update Model Size Tables
I wrote a Bash script that you can find here:
* [download\_and\_extract\_tf\_models.sh](./src/models/download_and_extract_tf_models.sh)

It goes over the following list of models, downloads them, untars, and removes lingering
tar file, while recording proxies for model size along the way (how long the model takes
to download, how long it takes to untar, how big the model directory is, and ultimately
how big the frozen graph is).

```
# Model URL Suffixes
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
```

(Since I was creating a MarkDown table, maybe I should have converted the 
underscores in the model names:  `s/_/\\_/g`)

The untarred models can be found in [models/tensorflow](./models/tensorflow).

This script outputs the following table, which can be found in the log 
file @ [logs/model\_size\_proxies.md](./models/model_size_proxies.md). 



|  Model Name             |  Wget Time | Tar Time | Dir Size | Graph Size |
|-------------------------|------------|----------|----------|------------|
| ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03 | 9.46  |0.26  |55M  |18M  |
| ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03 | 8.59  |0.18  |38M  |10M  |
| ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 | 24.77  |0.78  |148M  |49M  |
| ssdlite_mobilenet_v2_coco_2018_05_09 | 13.01  |0.30  |60M  |19M  |
| ssd_mobilenet_v2_coco_2018_03_29 | 60.07  |1.05  |201M  |66M  |
| ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 | 117.68  |2.07  |386M  |128M  |
| ssd_inception_v2_coco_2018_01_28 | 100.71  |1.69  |294M  |97M  |
| faster_rcnn_inception_v2_coco_2018_01_28 | 37.89  |0.89  |166M  |55M  |
| faster_rcnn_resnet50_lowproposals_coco_2018_01_28 | 119.12  |2.19  |405M  |115M  |
| faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 | 189.88  |3.94  |727M  |239M  |
| faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28 | 158.16  |3.46  |740M  |251M  |
| faster_rcnn_nas_lowproposals_coco_2018_01_28 | 268.90  |6.43  |1.2G  |405M  |


-----------------------------------------------------


## A Note on Running Locally

The servers herein are configured to utilize the Udacity classroom workspace. As such,
to run on your local machine, you will need to change the below file:

```
webservice/ui/src/constants/constants.js
```

The `CAMERA_FEED_SERVER` and `MQTT_SERVER` both use the workspace configuration. 
You can change each of these as follows:

```
CAMERA_FEED_SERVER: "http://localhost:3004"
...
MQTT_SERVER: "ws://localhost:3002"
```

-------------------------------------------------------------



