# LaneMarkingDetectionAssessment - Silent Testing
This repo aims for evaluating a lane marking detection algorithm based on the Silent Testing concept.

**Setup**:
* Creating a conda environment, and install the `requirements.txt`.
* Download the [NuScene](https://nuscenes.org/nuscenes#download) Dataset and also install the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).
* The dataset and the devkit should be put at the same level of directory of the main code.

**Run**
* using the `GetResolution.ipynb` to get the pixel per meter in x and y directions.
* Enter the above values in `LaneDetection.py`, which also need the source points as input.
* You can use the `FindScene.py` to find the scene and sample id given an image name

**Adversarial images**
The code to generate adversarial images from large-scale open-sourced datasets can be found [here](https://github.com/ADSafetyJointLab/PBLMD).
