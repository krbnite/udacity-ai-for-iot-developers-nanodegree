
For some reason, I cannot get NCS2 to work on my MacBook Pro.  

```
opt/intel/openvino/deployment_tools/demo/demo_squeezenet_download_convert_run.sh -d MYRIAD
```

I get an error:  `Can not init Myriad device: NC_ERROR`

I've googled it.  People have had similar issues, but the Intel support on the 
support forums haven't been too helpful.

Many of the docs
pages that discuss NCS2 only talk about Windows and Linux, so there is a 
possibility that is simply doesn't work on MacOS... Yet, there have been
multiple updated versions of OpenVINO and the corresponding docs since I
have first had this problem, and the docs page for MacOS still lists the
same ambiguous advice for using NCS2.  Though I haven't updated my OpenVINO 
(which is the 2019.3.376 version), from the docs and the few forums I've
found on this issue, it SHOULD WORK.


# Toying w/ Linux Dockers on top of MacOS
A while back, I made a Linux Dockerfile for an OpenVINO environment, and the
corresponding Docker Image is still on my computer.  

Seems like when I spin up one of my openvino dockers, at minimum I have to:

```
sudo apt-get update
sudo apt-get install vim
sudo apt-get install udev
opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

This should have been included in the Dockerfile!  Oh well, doesn't matter
anyway -- still can't get NCS2/MYRIAD to work.


# Tinkering w/ Intel's Dockers: Deep Learning WorkBench & More
Figured: "Hey, I'm not a Docker pro.  Maybe my Dockerfile is simply inadequate."

So, I looked for some official releases from Intel, and stumbled across their
WorkBench.  

```
# Get WorkBench
wget https://raw.githubusercontent.com/openvinotoolkit/workbench_aux/master/start_workbench.sh

# Give it executable permissions
chmod +x start_workbench.sh
```

After waiting for all the Docker Image dependencies to download, starting up 
WorkBench was as simple as:

```
./start_workbench.sh    # for default start-up
```

The more explicit start-up is:
```
./start_workbench.sh -IMAGE_NAME openvino/workbench
```

I suppose this is in case you changed the Image's name, or maybe customized it
and saved a new Image.

To use it with MYRIAD should be pretty dang simple:
```
./start_workbench.sh -IMAGE_NAME openvino/workbench -ENABLE_MYRIAD
```

Unfortunately, the assumption is that you are running on Ubuntu Linux for this 
to work.  I got the following error:
```
docker: Error response from daemon: Mounts denied: 
The path /dev/bus/usb
is not shared from OS X and is not known to Docker.
You can configure shared paths from Docker -> Preferences... -> File Sharing.
See https://docs.docker.com/docker-for-mac/osxfs/#namespaces for more info.
```

If I had read about the WorkBench a little more before I downloaded it, I would 
have known this!


For MacOS, the WorkBench documentation does not even recommend using 
`./workbench` command line utility. Instead, the documentation suggests that a 
MacOS user manually spins up the docker like so: 
```
docker run -p 127.0.0.1:5665:5665 \
    --name workbench \
    --volume ~/.workbench:/home/openvino/.workbench \ 
    -it openvino/workbench:latest
```

IMHO, they should also include the `--rm` flag since there is usually no good
reason to have the image being preserved after you're done with the Docker
session...  The script takes care of this for you:  it doesn't use `--rm`, but
it does first look to see if there is already a WorkBench image and deletes it
if it exists.  

I looked into other Intel Dockers before I quit this route:  all of them state
that they only support CPU on MacOS, whereas they support MYRIAD for Windows 
and Ubuntu.

For example:
* [openvino/ubuntu18_dev](https://hub.docker.com/r/openvino/ubuntu18_dev)
* > "If your host machine is MacOS then inference run inside the docker image 
  > is available for CPU only."


# Is MYRIAD recognized on my MacBook?

I explored this question a while back and had found a nice little utility 
called `lsusb`, which lists the various USB devices on your MacOS.  But this
time around on some Intel forum, an Intel support person recommended using
`system_profiler SPUSBDataType`, which I did:

```
system_profiler SPUSBDataType
    USB:
    
        USB 3.0 Bus:
    
          Host Controller Driver: Apple<DriverID>
          PCI Device ID: <DeviceID>
          PCI Revision ID: <RevisionID>
          PCI Vendor ID: <VendorID>
    
            Apple Internal Keyboard / Trackpad:
    
              Product ID: <ProductID>
              Vendor ID: <VendorID> (Apple Inc.)
              Version: <Version>
              Serial Number: <SerialNum>
              Speed: Up to 12 Mb/sec
              Manufacturer: Apple Inc.
              Location ID: <LocationID>
              Current Available (mA): 500
              Current Required (mA): 500
              Extra Operating Current (mA): 0
              Built-In: Yes
    
            Bluetooth USB Host Controller:
    
              Product ID: <ProductID>
              Vendor ID: <VendorID> (Apple Inc.)
              Version: <Version>
              Manufacturer: Broadcom Corp.
              Location ID: <LocationID>
```

Importantly, MYRIAD was NOT listed.  

I figured I'd try `lsusb` as well:


```
lsusb
    Bus <BusNum> Device <DeviceNum>: ID <ID> Apple Inc. Apple Internal Keyboard / Trackpad Serial: <SerialNum> 
    Bus <BusNum> Device <DeviceNum>: ID <ID> Apple Inc. Bluetooth USB Host Controller 
    Bus <BusNum> Device <DeviceNum>: ID <ID> Linux Foundation USB 3.0 Bus 
```

Yep: MYRIAD not listed.

The forum suggested removing and reinserting the NCS2 into another USB port 
(maybe the one in use is broken or corrupted somehow), which I did.  It seemed
to work, if only for a little while:


```
lsusb
    Bus <BusNum> Device <DeviceNum>: ID <ID> Apple Inc. Apple Internal Keyboard / Trackpad Serial: <SerialNum> 
    Bus <BusNum> Device <DeviceNum>: ID <ID> Apple Inc. Bluetooth USB Host Controller 
    Bus <BusNum> Device <DeviceNum>: ID <ID> 03e7 Movidius MyriadX  Serial: <SerialNum>
    Bus <BusNum> Device <DeviceNum>: ID <ID> Linux Foundation USB 3.0 Bus 
```


I tried running one of the MYRIAD demos again, but again it failed.  However,
this time there was  new error:
```
E: [ncAPI] [    375089] [] ncDeviceOpen:859 Device doesn't appear after boot
```

Interesting... Let's run it again!  This time the new error did not appear, so
I ran `lsusb` again to find the MYRIAD was no longer a recognized device.  

"Ok," I thought, "So basically, so I removed the NCS2, then reinserted it in a
new USB port -- and the system was able to recognize it.  However, at some point 
during running the demo, it lost track of the device and crashed.  The NCS2
after this point could no longer be 'seen' by my MacBook."

It occurred to me the original USB port was probably fine.  To test this out, 
I did everything all over:  removed the NCS2 from the second USB port,
reinserted it into the original USB port, ran the demo once, then ran the demo 
again.  The same behaviors presented themselves!  

1. Yes, MYRIAD was indeed recognized at first.
2. Yes, after running the model and getting the 'new' error, MYRIAD was no longer 
   recognized.
3. Yes, when running the model a second time, this 'new' error was gone (just the
   original error I was curious about remained).
4. Yes, the system no longer recognized the NCS2.

In other words:  Both ports work just fine!   Something else is the problem.

# The Same Boat
This person on the [Intel support forums](https://forums.intel.com/s/question/0D50P00004TZVlf/error-macos-can-not-init-myriad-device-ncerror?language=en_US) 
had the same exact issue. They dealt with Intel support for a couple
months, but ultimately to no avail.  The back-and-forth was somehwat repetitive:
* Intel just kept saying (paraphrasing here):  We tried it on multiple Mac systems; 
  it worked for us.  Not sure what your problem is.
* Developer with this issue would respond (again, paraphrasing):  Yes, I’ve tried it 
  on multiple Mac systems too; didn’t work on any.  I've changed ports.  I've made
  sure its USB3.0, and so on.

One of the things the support associate said, basically as a last ditch effort,
was that maybe the developer should uninstall and reinstall OpenVINO -- maybe
something got corrupted somehow.

So I figured that's what I'd try next.

# Uninstalling & Reinstalling OpenVINO

To uninstall openvino:
```
open /opt/inte/openvino/openvino_toolkit_uninstaller.app
```

I originally had OpenVINO version `2019.3.376`.  The new version that I've just
downloaded from the site is `2020.3.194`.

One of the quickest screw-ups I made after doing the basic install was to then
run the Model Optimizer's dependency installer script in my `base` Anaconda
environment -- my main environment that I use for most of my work-related 
development!  This is a HUGE no-no: don't do it.  Make sure to create
and/or activate an OpenVINO-specific Conda environment.  Or else the curse-ed 
installer will find and replace all of your TensorFlow and Numpy related python
libraries:  I had TF 2.2 and a bunch of the latest versions of associated packages,
like `tensorflow-estimators` and `tensorflow-probability`.  The installer replaced
TF 2.2 with TF 1.15, and other horrifying changes.

I uninstalled OpenVINO completely again.  Pip uninstalled anything related to 
TensorFlow and re-built my environment... 

```
pip freeze | grep tensor

sudo pip uninstall tensorflow tensorboard tensorflow-estimator tensorflow-probability tensorflow-datasets tensorflow-hub tensorflow-metadata tensorboard-plugin-wit
 
# .........

pip install tensorflow
pip install tensorboard
pip install tensorflow-estimator
pip install tensorflow-probability
pip install tensorflow-datasets
pip install tensorflow-hub
pip install tensorflow-metadata
pip install tensorboard-plugin-wit
```

Then I re-installed OpenVINO and did things the right way in my 
OpenVINO-specific `py35` environment (yes, I already had this made, which 
made my mistake all the more frustrating).


# Getting the SqueezeNet Demo to Work (at all)

I've had to do this a few times already (after each re-install), so might 
as well document it:
* To get the squeezenet demo script to work, I have to manually edit the bash script 
  to use the right version of pip.
  ```
  sudo vim /opt/intel/openvino/deployment_tools/demo/demo_squeezenet_download_convert_run.sh
  ```
* Near line 125-150:  the bash script checks if the MacOS-related system 
  (DARWIN) has python3.7 
  - if it doesn't, it moves onto python3.6 and so on
* Despite strictly using Anaconda and specifically being in a python3.5 Conda
  environment, my system was getting flagged for having python3.7
  - the script then tried to use `pip3.7`, which I don’t have -- thus the script crashes 
  - unfortunately, though I never use it, there is a copy of python3.7 laying around far 
    down my PATH variable (I went to remove it but got warned about dependencies) 
  - So, editing the bash script was my simplest solution
* Solution
  ```
  # comment out relevant block of code
  pip_binary=`which pip`
  python_binary=`which python`
    ```


# MYRIAD: Recognized

Well, happy to say that uninstalling and re-installing worked.
