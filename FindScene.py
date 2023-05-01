from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyquaternion import Quaternion
import utils
import matplotlib
from nuscenes.utils.geometry_utils import view_points

camera_name = ["CAM_FRONT"]
root_path = 'D:/Work/nuscene/data/sets/nuscenes/'
nusc = NuScenes(version='v1.0-trainval', dataroot='D:/Work/nuscene/data/sets/nuscenes', verbose=True)


def get_samples(scene_id):
    '''get samples in a given scene'''
    current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
    samples = []
    while not current_sample["next"] == "":
        samples.append(current_sample)
        # Update the current sample with the next sample
        current_sample = nusc.get('sample', current_sample["next"])
    return samples


def get_img_info(sample):
    # read the front camera info
    cam_token = sample['data'][camera_name[0]]
    cam_front_data = nusc.get('sample_data', cam_token)
    camera_calibration = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

    return cam_front_data, camera_calibration


if __name__ == '__main__':
    scene_found = False
    for id, scene in enumerate(nusc.scene):
        samples = get_samples(id)
        for idx, sample in enumerate(samples):
            cam_front_data, camera_calibration = get_img_info(sample)
            if cam_front_data["filename"] == 'samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151693612404.jpg':
                print(scene)
                print(idx)
                print(id)
                scene_found = True
                break
        if scene_found:
            break