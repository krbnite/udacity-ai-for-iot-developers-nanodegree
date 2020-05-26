Video input
* Count number of people in frame
* Time each person spent in frame
* Send count to MQTT server

 


Provided Support
* Starter Code
    - inference.py
        * NEED TO WRITE CODE TO:
            - load intermediate representation (IR) of a model
            - perform inference using Inference Engine (IE)
    - main.py
        * NEED TO WRITE CODE TO:
            - Connect to MQTT server
            - Handle input stream
            - Call on inference.py to perform inference
            - Extract model output and draw necessary info on frame (bounding boxes, semantic masks, etc)
            - Perform analysis on output
            - Determine number of people in frame
            - Time spent in frame
            - Total number of unique people counted
            - Send stats to MQTT server
            - Send processed frame to FFMPEG server
* Server Stuff
    - MQTT Server script
        * Receives post-inference JSON from primary script (individuals in frame, duration they spent in frame, total people in frame)
    - UI Server script
        * Displays video feed and stats received from MQTT server
    - FFMPEG server script
        * Receives output images from main script (e.g., frames w/ bounding boxes, etc) and feeds them to UI server
 

Optional:  Can you save on network bandwidth by adding on/off toggle for sending/receiving image frames (only focus on stats from MQTT server when off)?

 

Write-Up
* Model performance before and after OpenVINO optimizations
* Discuss potential use cases of deployed people counter app

 

# Implementation
The nanodegree provides a virtual environment where you can do everything, 
but -- IMHO -- that just doesn't match the spirit of learning to run deep 
learning algorithms on edge devices!  


Fortunately, they also provide a
GitHub repo that can get you started on your laptop, etc, which I've added
as a Git Subtree:
* Original Repo:  [nd131: OpenVINO Fundamentals Project Starter](https://github.com/udacity/nd131-openvino-fundamentals-project-starter)


I recently purhcased the [CanaKit Rasberry Pi 4B 4GB Starter Pack](https://www.amazon.com/gp/product/B07V5JTMV9), 
which I plan to use in tandem with the Intel Neural Compute Stick 2 (NCS2) I bought a while back.

 

 

 

 

 
