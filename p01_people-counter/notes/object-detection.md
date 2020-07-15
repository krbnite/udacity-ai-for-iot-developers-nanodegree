bject detection is all about localizing (and usually classifying) any number 
of objects that an image may contain.  So, event/activity detection would be the
same:  given a time series, can we detect how many “objects” are in it (where 
"objects" are independent events and/or activities).  If I have 10 minutes of 
recording, can I draw bounding boxes around unique activities?  (This is a 
trickier problem than with images since so many activities can be superimposed, 
not just blocked/obscured from view.)

The naive approach is to cut up an image into a bunch of overlapped tiles and 
to classify each one.  However, you will find that it is tough to pick just one 
tile size:  objects in the image exist at different scales, so the best way to 
detect, localize, and classify all objects in an image is not to arbitrarily 
choose (or hyper-optimize) a single tile scale, but to tile the image at 
multiple scales.  But what about the overlap?  With too coarse a tiling scheme, 
you may find that an object is never really “front and center” in any of your 
tiles, making detection and classification difficult.  You’d likely find that a 
dense tiling scheme at each scale improves your model’s accuracy, but at the 
cost of many more tiles.  And even at this point, you will find that many 
objects are still not well-localized, e.g., a person can be captured and 
localized well vertically in a tile, but only take up the first 30% of the 
tile's horizontal axis.  Here, you would realize that it’s necessary to include 
some rectangular tiling schemes at each scale, i.e., it is important to cover a 
range of aspect ratios (width:height).  This can get compute-heavy pretty 
quickly!  Then there is the issue of counting distinct objects only once, e.g., 
you can imagine a person getting classified correctly in many of the tiles, 
across many aspect ratios and scales — how do you know if each of these is the 
same person or different people?


# Diving into Object Detection

Densely tiling an image at many scales and aspect ratios can quickly blow up.  
For example, consider a moderate-sized 300x300 image.  Let’s the largest scale 
is the image itself and the smallest scale we look at is 10 pixels.  We have to 
choose a list of scales:  10, 25, 50, 75, 100, 150, 200, 250, 300.  Then a list 
of aspect ratios, say: 1/3, 1/2, 1, 2, 3.  Then a density/overlap rule:  let's
say 80% per linear dimension.  And finally a padding at the edges: none (valid), 
same, something else?  We will choose “same padding” for now.  


At the 10-pixel 
scale, we have 5 tiling schemes:  (10,3), (10,5), (10,10), (3,10), (5,10).  80% 
overlap of a 10-pixel length gives a 2-pixel stride (1+(N-W)/S = 1 + (300-10)/2 
= 146 cuts).  For the 3- and 5-pixel lengths, 80% overlap rounds to 1-pixel 
strides (for 3 pixels, this is ceil(1+(300-3)/1)=298 cuts; for 5 pixels, this is 
ceil(1+(300-5)/1)=296 cuts.  So for our 5 tiling schemes then, we have 

```
TotalTiles = 146x298 + 146x296 + 146x146 + 296x146 + 298x146 
           = 2*43,508 + 2*43,216 + 21,316 
           =  194,764 
```

At just one scale, we are already almost 200k tiles.  Whoa.

At the 25-pixel scale, we have another 5 tiling schemes:  (25,8), (25,12), 
(25,25), (12,25), (8,25).  Here, we have 8-, 12-, and 25-pixel tile edges 
corresponding to axonal strides of 6, 8, and 20, and axonal tilings of 
* `ceil(1+(300-8)/2)=147`, 
* `ceil(1+(300-12)/4)=73`, and 
* `ceil(1+(300-25)/5)=56` 

This makes `2*56x147 + 2*56x73 + 56x56 = 27,776` tiles, bringing us to a total 
of `222,540` tiles.

Multiplying the scale by 2.5 (10 -> 25) gave us 14% of the tiles.  It’s likely 
to follow a similar rule when we go from 25 to 50, and so on — just small 
“correction terms” to our 222k estimate.  Maybe we get to 230k or so.  This is 
not a fully exhaustive search (where you would look over all scale sizes, 
positions, and aspect ratios), but it sure is exhaustive.  Maybe you might call 
it a nigh-exhaustive sliding window approach.  It’s likely to get decent 
results — at the cost of waiting forever.  Point is, that’s a lot of tiles 
to classify!

# R-CNN
One of the boons brought about by R-CNN is that it brought this kind of number 
down significantly — to 2000 tiles, or what they call regions.  That’s a much 
easier number to deal with.

* 2013: Girshick et al: [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

This paper holds some interesting historical notes: when the Krizhevsky [2012]
CNN paper hit the scene, its significance 
> "was vigorously debated during the ILSVRC 2012 workshop. The central
> issue can be distilled to the following: To what extent do
> the CNN classification results on ImageNet generalize to
> object detection results on the PASCAL VOC Challenge?"

Basically, people were like, "Ok, yea, this is good for a simple classification
task, but no way its going to generalize to object detection tasks." This 
is because the demands of detection require far more than classification. For
detection, is not good enough to say an object exists within an image: the
algorithm must be able to localize the object as well -- and usually not just
one, but likely many -- with no a priori idea how many.

The R-CNN paper stepped into the discussion 2 years later (2014) to shut the
case:  
> "We answer this question by bridging the gap between
> image classification and object detection. This paper is the
> first to show that a CNN can lead to dramatically higher object 
> detection performance on PASCAL VOC as compared
> to systems based on simpler HOG-like features."


One approach to localizing an object is through regression, e.g., learn 
to identify some centroid of the object. Another approach is to use sliding
windows, of which the tiling approach above is an example.  Sliding windows are
an "obvious" choice since they have been so successful historically speaking:
Fourier and wavelet analyses come to mind.  One can also consider the sliding
window at the classification layer, as is the case with CNNs.  However, for 
deep CNNs (anything with more than 2-3 layers, really), the receptive fields
(of the input layer) at or near the classification layer can be very large (not
 ideal for "localization").


To generate region proposals, the R-CNN papers uses Selective Search, which 
"which combines the strength of both an exhaustive search and segmentation. Like
 segmentation, we use the image structure to guide our sampling process. Like 
exhaustive search, we aim to capture all possible object locations. Instead of a
 single technique to generate possible object locations, we diversify our search
 and use a variety of complementary image partitionings to deal with as many 
image conditions as possible," as the authors of Selective Search say.

* 2012: Uijlings et al:  [Selective Search for Object Recognition](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)

The R-CNN paper says: 
* The R-CNN object detection system consists of three modules:  
  - **Region Proposal Generator**: The first R-CNN module "generates 
    category-independent region proposals.  These proposals define the set of 
    candidate detections available to our detector." They explain: "While R-CNN is 
    agnostic to the particular region proposal method, we use selective search 
    to enable a controlled comparison with prior detection work." 
  - **Pre-Trained, Fine-Tuned CNN Feature Extractor**:  
    The region proposals are then transformed into 
    227x227 RGB images, regardless of size or aspect ratio, so that they can be 
    fed into the feature extraction module of the R-CNN system. Again, the R-CNN
    framework is agnostic to the exact region transformation mechanism used to
    get the region proposals into the right shape for the CNN (the authors show
    a few other examples in the appendix), but the chosen mechanism for the paper 
    is a warping-with-context technique that adds a context frame around the 
    region proposal's bounding box such that the output 227x227 image has an
    8-pixel context frame around a 219x219 image (I discuss 
    this transformation in more detail in the upcoming excursion).  The feature
    extractor here is "a large convolutional neural network that 
    extracts a fixed-length feature vector from each region."   The CNN is
    a Caffe re-implementation of AlexNet,  pre-trained on an auxillary dataset
    (ILSVRC2012) using only image-level annotations (no bounding box info). For
    each dataset that the R-CNN system is benchmarked against for object 
    detection, the pre-trained CNN is fine-tuned: "To adapt our CNN to the new
    task (detection) and the new domain (warped proposal windows), we continue
    [SGD] training of the CNN parameters using only warped region proposals."
    The only surgery performed is trimming off AlexNet's 1000-unit 
    classification layer for an untrained (N+1)-unit classification layer 
    corresponding to a given target domain, where N is the number of object
    classes and the extra unit is for background classification (e.g., for
    VOC, N=20, and for ILSVRC2013, N=200). As a final note on the feature
    extraction module of R-CNN: the authors make another point that choosing
    AlexNet is somewhat arbitrary, and that it can be swapped out for another
    model, which they show using a 16-layer model (double the size of AlexNet).
      * SIDENOTE:  they refer to AlexNet as T-Net in the paper (for Toronto 
        Net), while the 16-layer networks is referred to as O-Net (for Oxford
        Net), though it more commonly known as VGG these days (VGG stands for
        the Visual Geometry Group at Oxford).
  - The third module of the R-CNN system "is a set of class-specific linear SVMs."

* "At test time, our method generates around 2000 
  category-independent region proposals for the input image, extracts a 
  fixed-length feature vector from each proposal using a CNN, and then classifies 
  each region with category-specific linear SVMs. We use a simple technique 
  (affine image warping) to compute a fixed-size CNN input from each region 
  proposal, regardless of the region's shape."


## Excursion:  Region Proposal Context Warping
I figured out the various linear algebraic transformations required to do the
context warping they discuss.  Basically, the algorithm is like this:
1 . Collect input data  
  - region proposal `(xL, xR, yB, yU)` 
  - output height and width, `h_out` and `w_out` (e.g., 227x227 for R-CNN paper)
  - output width and height of context frame, `w_frame` and `h_frame`
    (e.g., the R-CNN paper calls for 16 context pixels of height and width, or 
    what I refer to as an 8-pixel frame width and height)
2. Compute the affine transformation that maps the region proposal to the frame
   interior region 
  - compute region proposal width and height, matrix coeffs a11 and a22,
    and the transformation offsets b1 and b2
  - in general the transformation looks like:
    ```
    T:(xL,xR,yB,yU) -> (w_frame, w_out - w_frame, w_frame, h_out - w_frame)
    ```
  - e.g., in the R-CNN paper, this would look like:
    ```
    T:(xL,xR,yB,yU) -> (w_frame, 227 - w_frame, w_frame, 227 - w_frame)
    ```
3. Find the (x,y) coordinates of bottom left and top right corners of the
   context frame around the region proposal in the original image space that
   map to `(0,0)` and `(w_out,h_out)` (i.e., that creates a `w_out x h_out` 
   warped image)
   

I've worked out some corresponding python code.  It doesn't actually transform
images, but just goes through the steps of figuring out what the input context
region should be -- so you know how to cut the image around the region 
proposal to submit for warping.

```python
def warped_context_region(
    region,
    context = 16,
    h_out = 227,
    w_out = 227,
):
    """
    INPUTS
      region: 4-tuple: (xL, xR, yB, yU)
      context: integer:  total pixel padding along output axes; padding
          is evenly distributed at both sides of an axis (e.g., a 16-pixel 
          context gives a 8-pixel context frame around the warped region 
          proposal)
    
    OUTPUTS
       Input context region: 4-tuple (cxL, cxR, cyB, cyU)
           The region in the image to warp so that the warped image has
           the chosen context frame.
    """ 
    xL, xR, yB, yU = region
    # Input Width & Height
    w_in = xR - xL
    h_in = yU - yB
    # Transformation Coefficients
    a11 = (w_out - context)/w_in
    a22 = (h_out - context)/h_in
    b1  = np.ceil(context/2) - a11 * xL
    b2  = np.ceil(context/2) - a22 * yB
    # Input Context Region
    delta_x = np.round((context/2) * w_in / (w_out - context))
    delta_y = np.round((context/2) * h_in / (h_out - context))
    cxL = xL - delta_x
    cyB = yB - delta_y
    cxR = xR + delta_x
    cyU = yU + delta_y
    return (cxL, cxR, cyB, cyU), \
           lambda x,y: (round(a11*x + b1), round(a22*y + b2))
```


## Back to R-CNN
#### Pre-trained models as feature extractors
For the pre-trained CNN without fine-tuning, they found that using the output
of the 5th layer (last conv-pool layer) worked ok as features.  However, the
output of the subsequent FC layer provided much better features.  Importantly,
the output of the next FC layer provided features that degraded the performance.
The moral here is that going to deep into a pre-trained model (without any
subsequent fine-tuning) can land you in a space too specific to the original
training: the generalization of the outputs declines as one goes too deep.

#### The powerful convolution and the lackluster FC layer
Though the features provided by the first FC layer bumps the performance, this
bump is ever so slight: from 44.2 mAP to 46.2 mAP.  Now consider the number of
parameters: cutting off at the last convolutional layer utilizes only 6% of the
total network's parameters (about 3.5M parameters), while including the first 
FC layer introduces another 65% of the network's parameter count (about 37.7M 
parameters).  That's crazy!  What would be the performance boost of an extra
convolutional layer instead of that first FC layer?  It's possible FC layers
aren't really needed -- at the least, it's reasonable to conclude that most of
the FC layer does not matter (maybe one can randomly assign 0 weights that do
not train).

#### Pre-trained, fine-tuned models as feature extractors
When fine-tuning a pre-trained model, the R-CNN authors found that the FC
layers strongly come back into play.  The last convolutional layer gets about
a 3-point mAP bump, from 44.2 to 47.3.  However, the first FC layers bumps
about 7 points from 46.2 to 53.1, while the last FC layers bumps 9.5 points
from 44.7 to 54.2.  In other words, the fine-tuning somewhat enhanced the 
convolution, but mostly served to enhance the subsequent FC layers, giving
some insight: the conv layers learn generalities while the FC layers learn 
specifics.  Here we see that going deeper no longer degrades the provided 
features since the FC layers have specialized to the new dataset.

#### Deeper Feature Extractors
To highlight R-CNN's system-oriented, modular design (in contrast to it being
a specific architecture), the authors swap out AlexNet for a 16-layer VGGNet. 
The mAP increases fairly dramatically from 58.5% to 66.0%.  However, the
compute time takes a 7x hit -- so it's a true tradeoff space (e.g., on a 
low-risk mobile application, probably want to go w/ the AlexNet version)...at
least in 2014 when the paper came out.

------------------------------------------


# Fast R-CNN

In the day of end-to-end deep learning solutions, it is interesting to learn 
about R-CNN's modular system.  However, it should also be obvious at this point
what changes one might attempt to improve the system:
* the selective search algorithm does not learn or adapt to a new domain; it 
  would be helpful if the mechanism that generates region proposals could learn
  to do so more accurately and efficiently; the proposal generator's ability to
  accurately identify all regions of interest can be a bottleneck in this 
  framework since if an object is not detected in this stage (isolate within a
  proposal), then it cannot be classified downstream (however, generating too
  many proposals bottlenecks the speed and creates a highly imbalanced dataset
  of proposals, most of them being 'background', so it would be best to learn
  an accurate, efficient proposal mechanism)
* the warping mechanism is inefficient: it needs to be computed for each region
  proposal separately; the warping mechanism is also static: it does not learn
  or adapt to the data domain
* the SVM at the end learns classifications, but this knowledge is trickled 
  back into the feature extractor (or proposal generator), which is a loss of
  cohesion between modules

The R-CNN author, Girshick, was well aware of all this, and worked to improve
R-CNN over time -- documented in a sequence of papers on the system.

The first of these issues that was tackled is the warping mechanism: in the
Fast R-CNN system, it is swapped out for an RoI pooling layer.

The faster R-CNN paper addresses the selective search issue: it swaps out
the region proposal generator with the region proposal network (RPN), which
is a branch of the full network (end-to-end proposals, warping, learning,
and classification).


Some refs to look at or watch:
* https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
* https://d2l.ai/chapter_computer-vision/rcnn.html
* https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html
  - they always have good docs

