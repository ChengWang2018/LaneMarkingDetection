from nuscenes.nuscenes import NuScenes
import os

def get_samples(scene_id: int):
    '''get samples in a given scene'''
    current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
    samples = []
    while not current_sample["next"] == "":
        samples.append(current_sample)
        # Update the current sample with the next sample
        current_sample = nusc.get('sample', current_sample["next"])
    return samples


def get_img_info(sample):
    '''read the front camera info'''
    cam_token = sample['data'][camera_name[0]]
    cam_front_data = nusc.get('sample_data', cam_token)
    camera_calibration = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

    return cam_front_data, camera_calibration


if __name__ == '__main__':
    ''' This function is used to find the scene id and sample id given the image name'''

    camera_name = ["CAM_FRONT"]
    # get the current working directory
    current_path = os.getcwd()
    parent = os.path.dirname(current_path)
    root_path = parent + '/data/sets/nuscenes/'

    nusc = NuScenes(version='v1.0-trainval', dataroot='D:/Work/nuscene/data/sets/nuscenes', verbose=True)

    scene_found = False
    # Given the image name that you are interested in
    img_name = 'n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385098862404.jpg'
    for id, scene in enumerate(nusc.scene):
        samples = get_samples(id)
        for idx, sample in enumerate(samples):
            cam_front_data, camera_calibration = get_img_info(sample)
            if cam_front_data["filename"] == 'samples/CAM_FRONT/' + img_name:
                print(scene)
                print(idx)
                print(id)
                scene_found = True
                break
        if scene_found:
            break
