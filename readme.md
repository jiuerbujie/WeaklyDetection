# Code for Weakly Supervised Object Detection

## Installation
1. Clone this repository by :
```Shell
git clone --recursive https://github.com/jiuerbujie/WeaklyDetection
```
Assume the top-level directory is ```ROOT```.

2. Install caffe under ```ROOT/py-faster-rcnn/caffe-fast-rcnn```. Both Matlab and Python interfaces are required.

3. Download VOC dataset [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/) under ```data``` folder.

    3.1 Download VOC2007 into ```ROOT/data/VOCdevikit/VOC2007```.

    3.2 Download VOC2012 into ```ROOT/data/VOCdevikit/VOC2012```.

4. Download the pre-trained models. See ```ROOT/models/Readme.md``` for all pre-trained models.

5. Test models by running ```test_matlab.m```. Modify the following lines:
```Matlab
imdb = ...
exp_name = ...
netdef = ...
model = ...
```
It will save detection results into a  ```_dets.mat``` file.

6. Evaluate the model by running ```eval_test.m```. It generates a txt file consists of mAPs in ```exp/$exp_name``` folder.