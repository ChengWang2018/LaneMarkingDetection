import numpy as np
from shapely.geometry import Polygon
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from clip_points_behind_camera import clip_points_behind_camera
from nuscenes.utils.geometry_utils import view_points
import cv2


def world2pixel(points):
    points = np.vstack((points, np.zeros((1, points.shape[1]))))
    # Transform into the ego vehicle frame for the timestamp of the image.
    points = points - np.array(ego_pos['translation']).reshape((-1, 1))
    points = np.dot(Quaternion(ego_pos['rotation']).rotation_matrix.T, points)
    # Transform into the camera.
    points = points - np.array(cs_record['translation']).reshape((-1, 1))
    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(points, cam_intrinsic, normalize=True)

    return points


def pixel2world(new_points_pixel, cam_intrinsic):
    r_camera2body = Quaternion(cs_record['rotation']).rotation_matrix.T
    t_camera2body = np.array(cs_record['translation']).reshape((-1, 1))
    trans_camera2body = np.hstack((r_camera2body, t_camera2body))
    trans_camera2body = np.vstack((trans_camera2body, np.array([0, 0, 0, 1])))

    r_body2world = Quaternion(ego_pos['rotation']).rotation_matrix.T
    t_body2world = np.array(ego_pos['translation']).reshape((-1, 1))
    trans_body2world = np.hstack((r_body2world, t_body2world))
    trans_body2world = np.vstack((trans_body2world, np.array([0, 0, 0, 1])))

    cam_intrinsic = np.hstack((cam_intrinsic, np.array([0, 0, 0]).reshape(-1, 1)))

    trans_camera2world = np.dot(trans_camera2body, trans_body2world)
    trans_camera2world = np.dot(cam_intrinsic, trans_camera2world)

    Pp = np.linalg.pinv(trans_camera2world)
    X = np.dot(Pp, new_points_pixel)
    X1 = np.array(X[:3], float) / X[3]

    return X1


def pixel2world_new(new_points_pixel, cam_intrinsic):
    point = np.dot(np.linalg.pinv(cam_intrinsic), new_points_pixel)
    points = np.dot(np.linalg.pinv(Quaternion(cs_record['rotation']).rotation_matrix.T), point)

    transPlaneToCam = np.dot(np.linalg.pinv(Quaternion(cs_record['rotation']).rotation_matrix.T), cs_record['translation']).reshape((-1, 1))
    scale = -transPlaneToCam[2] / points[2]

    scale_points = np.multiply(scale, points)
    body_points = scale_points + transPlaneToCam


    transPlaneToWorld = np.dot(np.linalg.pinv(Quaternion(ego_pos['rotation']).rotation_matrix.T), ego_pos['translation']).reshape((-1, 1))
    world_points = np.dot(np.linalg.pinv(Quaternion(ego_pos['rotation']).rotation_matrix.T), body_points)

    world_points = world_points + transPlaneToWorld

    return body_points


def pixel_to_world(img_points, cam_intrinsic):

    camera_intrinsics = cam_intrinsic
    r = Quaternion(cs_record['rotation']).rotation_matrix.T
    t = np.array(cs_record['translation']).reshape((-1, 1))
    K_inv = np.linalg.inv(camera_intrinsics)
    R_inv = np.linalg.inv(r)
    R_inv_T = np.dot(R_inv, t)
    world_points = []
    coords = img_points

    cam_point = np.dot(K_inv, coords)
    cam_R_inv = np.dot(R_inv, cam_point)
    scale = R_inv_T[2] / cam_R_inv[2]
    scale_world = np.multiply(scale, cam_R_inv)
    world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
    pt = np.zeros((3, 1), dtype=np.float64)
    pt[0] = world_point[0]
    pt[1] = world_point[1]
    pt[2] = 0
    world_points.append(pt.T.tolist())

    return world_points


if __name__ == '__main__':
    dataroot = 'D:/Work/nuscene/data/sets/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    scene_id = 3
    sensor = 'CAM_FRONT'

    samples = []
    current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
    while not current_sample["next"] == "":
        samples.append(current_sample)
        # Update the current sample with the next sample
        current_sample = nusc.get('sample', current_sample["next"])

    current_sample = samples[14]  # chose one for an example
    cam_token = current_sample['data'][sensor]
    cam_front_data = nusc.get('sample_data', cam_token)
    ego_pos = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
    cs_record = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    cam_path = nusc.get_sample_data_path(cam_token)

    points_world = np.array([1817.779, 865.6015], float).reshape(2, 1)
    points_pixel = world2pixel(points_world)
    print('original point is: ', points_world)
    print('converted_points is: ', points_pixel)

    new_points_pixel = np.array([652.703, 326.3675, 1], float).reshape(-1, 1)
    new_points_world = pixel2world_new(new_points_pixel, cam_intrinsic)
    print('original pixel point is: ', new_points_pixel)
    print('converted world point is: ', new_points_world)

    new_points_pixel = np.array([652.703, 326.3675, 1], float).reshape(-1, 1)
    new_points_world_ = pixel_to_world(new_points_pixel, cam_intrinsic)
    print('original pixel point is: ', new_points_pixel)
    print('converted world point is: ', new_points_world_)
