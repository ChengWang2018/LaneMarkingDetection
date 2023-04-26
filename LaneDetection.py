from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyquaternion import Quaternion
import utils
import matplotlib

matplotlib.use('TkAgg')


class LaneDetection:
    def __init__(self):
        self.right_fit_m = None
        self.left_fit_m = None
        self.right_fit = None
        self.left_fit = None
        self.ym_per_pix = 0.2873
        self.xm_per_pix = 0.02555

        # Point order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        self.source_points = np.array([[680, 565], [970, 565], [1350, 900], [190, 900]], dtype="float32")
        self.bottom_width = self.source_points[2, 0] - self.source_points[3, 0]
        self.height = self.source_points[2, 1] - self.source_points[0, 1]

        self.Minv, self.M = self.get_perspective_matrix()

    def get_perspective_matrix(self):
        destination_points = np.array([
            [0, 0],
            [self.bottom_width - 1, 0],
            [self.bottom_width - 1, self.height - 1],
            [0, self.height - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(self.source_points, destination_points)
        Minv = cv2.getPerspectiveTransform(destination_points, self.source_points)

        return Minv, M

    def perspective_transform(self, img):
        image_unwarped = cv2.warpPerspective(img, self.M, (int(self.bottom_width), int(self.height)))
        return image_unwarped

    @staticmethod
    def draw_polygon_on_image(source_points, img, line_color=(0, 255, 0)):
        points_num = len(source_points)
        for inx in range(points_num):
            if inx < points_num - 1:
                p1 = (int(source_points[inx][0]), int(source_points[inx][1]))
                p2 = (int(source_points[inx + 1][0]), int(source_points[inx + 1][1]))
            else:
                p1 = (int(source_points[inx][0]), int(source_points[inx][1]))
                p2 = (int(source_points[0][0]), int(source_points[0][1]))
            cv2.line(img, p1, p2, line_color, 3)

        return img

    @staticmethod
    def hist(img):
        """
        This is used to extract data points for a histogram
        """
        # Grab only the bottom half of the image

        bottom_half = img[img.shape[0] // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    @staticmethod
    def threshIt(img, thresh_min, thresh_max):
        """
        Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
        """
        xbinary = np.zeros_like(img)
        xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
        return xbinary

    def absSobelThresh(self, img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
        """
        Calculate the Sobel gradient on the direction `orient` and return a binary thresholded image
        on [`thresh_min`, `thresh_max`]. Using `sobel_kernel` as Sobel kernel size.
        """
        if orient == 'x':
            yorder = 0
            xorder = 1
        else:
            yorder = 1
            xorder = 0

        sobel = cv2.Sobel(img, cv2.CV_64F, xorder, yorder, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))
        return self.threshIt(scaled, thresh_min, thresh_max)

    @staticmethod
    def combineGradients(sobelX, sobelY):
        """
        Compute the combination of Sobel X and Sobel Y or Magnitude and Direction
        """
        combined = np.zeros_like(sobelX)
        combined[((sobelX == 1) & (sobelY == 1))] = 1
        return combined

    def calculateCurvature(self, yRange, left_fit_cr):
        """
        Returns the curvature of the polynomial `fit` on the y range `yRange`.
        """
        return ((1 + (2 * left_fit_cr[0] * yRange * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])

    def drawLine(self, img):
        """
        Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
        """
        yMax = int(self.height)
        ploty = np.linspace(0, yMax - 1, yMax)
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Calculate points.
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0]))
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    def findLines(self, binary_warped, nwindows=9, margin=110, minpix=50):
        """
        Find the polynomial representation of the lines in the `image` using:
        - `nwindows` as the number of windows.
        - `margin` as the windows margin.
        - `minpix` as minimum number of pixes found to recenter the window.
        - `ym_per_pix` meters per pixel on Y.
        - `xm_per_pix` meters per pixels on X.

        Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Fit a second order polynomial to each
        self.left_fit_m = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        self.right_fit_m = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        # return (
        #     left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)

    def get_left_right_lane(self, ploty):
        lineLeft = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        lineRight = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        return lineLeft, lineRight

    def pixel2world(self, img, sensor_calibration, ego_pos):
        ''' Tranform pixel coordinates to the vehicle body frame '''
        lanes_on_vehicle_coords = []

        yMax = int(self.height)
        ploty = np.linspace(0, yMax - 1, yMax)
        lineLeft, lineRight = self.get_left_right_lane(ploty)

        # get body to world
        rvec_body2world = Quaternion(ego_pos['rotation']).rotation_matrix.T
        tvec_body2world = np.array(ego_pos['translation']).reshape((-1, 1))

        # get camera to body
        rvec_camera2body = Quaternion(sensor_calibration['rotation']).rotation_matrix.T
        tvec_camera2body = np.array(sensor_calibration['translation']).reshape((-1, 1))

        # get pixel to camera
        intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])

        for lane in [lineLeft, lineRight]:
            lane_on_vehicle_coords = []
            for u, v in zip(lane, ploty):
                # get the point on original image
                persepctive_points = np.array([u, v], dtype=float)
                point = cv2.perspectiveTransform(persepctive_points.reshape(-1, 1, 2), self.Minv)

                point = np.array([point[0][0][0], point[0][0][1], 1], float).reshape(-1, 1)
                point = np.dot(np.linalg.pinv(intrinsic_matrix), point)

                point = np.dot(np.linalg.pinv(rvec_camera2body), point)
                point = point + tvec_camera2body

                # point = np.dot(np.linalg.pinv(rvec_body2world), point)
                # point = point + tvec_body2world

                lane_on_vehicle_coords.append(point)
            lanes_on_vehicle_coords.append(lane_on_vehicle_coords)

        return lanes_on_vehicle_coords

    def body2map(self):
        '''Tranform body frame to map frame'''
        pass

    def display_offset(self, lanes_on_vehicle_coords, new_img, fontScale=2):
        '''display offset info on the image'''
        # unclear how it is defined
        yRange = 719

        # Calculate curvature
        leftCurvature = self.calculateCurvature(yRange, self.left_fit_m)
        rightCurvature = self.calculateCurvature(yRange, self.right_fit_m)

        # Calculate vehicle center
        left_line_pos = lanes_on_vehicle_coords[0][-1][0][0]
        right_line_pos = lanes_on_vehicle_coords[1][-1][0][0]
        diffFromVehicle = (right_line_pos + left_line_pos) / 2
        if diffFromVehicle > 0:
            message = '{:.2f} m right'.format(diffFromVehicle)
        else:
            message = '{:.2f} m left'.format(-diffFromVehicle)

        # Draw info
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 255, 255)
        cv2.putText(new_img, 'Left curvature: {:.0f} m'.format(leftCurvature), (50, 50), font, fontScale, fontColor, 2)
        cv2.putText(new_img, 'Right curvature: {:.0f} m'.format(rightCurvature), (50, 120), font, fontScale, fontColor, 2)
        cv2.putText(new_img, 'Vehicle is {} of center'.format(message), (50, 190), font, fontScale, fontColor, 2)

        return new_img


class NuSceneProcessing:
    def __init__(self, camera_name, scene_id, root_path):
        self.camera_name = camera_name
        self.scene_id = scene_id

        self.nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
        self.nusc_map = NuScenesMap(dataroot=root_path, map_name=self.get_map_name())

    def get_map_name(self):
        scene_record = self.nusc.scene[self.scene_id]
        log_record = self.nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        return log_location

    def get_samples(self):
        '''get samples in a given scene'''
        current_sample = self.nusc.get('sample', self.nusc.scene[self.scene_id]['first_sample_token'])
        samples = []
        while not current_sample["next"] == "":
            samples.append(current_sample)
            # Update the current sample with the next sample
            current_sample = self.nusc.get('sample', current_sample["next"])
        return samples

    def get_img_info(self, sample):
        # read the front camera info
        cam_token = sample['data'][self.camera_name[0]]
        cam_front_data = self.nusc.get('sample_data', cam_token)
        camera_calibration = self.nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])

        return cam_front_data, camera_calibration

    def get_ego_pos(self, cam_front_data):
        ego_pos = self.nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        return ego_pos

    def get_lane_on_map(self, ego_pos):
        closest_lane_token = self.nusc_map.get_closest_lane(ego_pos['translation'][0], ego_pos['translation'][1], radius=2)
        closest_lane = None
        try:
            closest_lane = self.nusc_map.get("lane", closest_lane_token)
        except KeyError as e:
            closest_lane_connector = self.nusc_map.get("lane_connector", closest_lane_token)

        point_xy = []
        # we skip lane connector
        if closest_lane is not None:
            poly = self.nusc_map.get("polygon", closest_lane['polygon_token'])

            for point in poly['exterior_node_tokens']:
                node = self.nusc_map.get('node', point)
                point_xy.append([node['x'], node['y']])

            point_xy = np.array(point_xy)
        return point_xy


def main(current_sample):
    '''The main process to handel the image data'''
    cam_front_data, camera_calibration = data.get_img_info(current_sample)

    cam_intrinsic = np.array(camera_calibration['camera_intrinsic'])
    image = cv2.imread(root_path + cam_front_data["filename"])

    # ego_pos
    ego_pos = data.get_ego_pos(cam_front_data)

    undistort_img = cv2.undistort(image, cam_intrinsic, None)
    hls_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2HLS)
    useSChannel = hls_img[:, :, 2]
    SobelX = lanedetection.absSobelThresh(useSChannel, thresh_min=10, thresh_max=160)
    SobelY = lanedetection.absSobelThresh(useSChannel, orient='y', thresh_min=10, thresh_max=160)

    resultCombined = lanedetection.combineGradients(SobelX, SobelY)

    img_unwarp = lanedetection.perspective_transform(resultCombined)
    lanedetection.findLines(img_unwarp)
    new_img = lanedetection.drawLine(image)

    # transform lane lines to vehicle body frame
    lanes_on_vehicle_coords = lanedetection.pixel2world(image, camera_calibration, ego_pos)
    gt = data.get_lane_on_map(ego_pos)
    utils.plot_lane_on_map(lanes_on_vehicle_coords, gt)
    input_img = lanedetection.display_offset(lanes_on_vehicle_coords, new_img)
    output = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    utils.showImages(output)
    plt.show()
    print('ok')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_name = ["CAM_FRONT"]
    scene_id = 3
    root_path = 'D:/Work/nuscene/data/sets/nuscenes/'
    data = NuSceneProcessing(camera_name, scene_id, root_path)
    lanedetection = LaneDetection()

    # to process a single image or all images in a scene
    single_img = True

    samples = data.get_samples()
    if single_img:
        current_samples = samples[0]
        main(current_samples)
    else:
        current_samples = samples
        for current_sample in current_samples:
            main(current_sample)

