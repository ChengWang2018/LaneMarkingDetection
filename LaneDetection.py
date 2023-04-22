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

    def draw_polygon_on_image(self, img, line_color=(0, 255, 0)):
        points_num = len(self.source_points)
        for inx in range(points_num):
            if inx < points_num - 1:
                p1 = (int(self.source_points[inx][0]), int(self.source_points[inx][1]))
                p2 = (int(self.source_points[inx + 1][0]), int(self.source_points[inx + 1][1]))
            else:
                p1 = (int(self.source_points[inx][0]), int(self.source_points[inx][1]))
                p2 = (int(self.source_points[0][0]), int(self.source_points[0][1]))
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

    def drawLine(self, img, left_fit, right_fit):
        """
        Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
        """
        yMax = int(self.height)
        ploty = np.linspace(0, yMax - 1, yMax)
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Calculate points.
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

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
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit a second order polynomial to each
        self.left_fit_m = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        self.right_fit_m = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        return (
            left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)

    def get_left_right_lane(self, ploty):
        lineLeft = self.left_fit_m[0] * ploty ** 2 + self.left_fit_m[1] * ploty + self.left_fit_m[2]
        lineRight = self.right_fit_m[0] * ploty ** 2 + self.right_fit_m[1] * ploty + self.right_fit_m[2]

        return lineLeft, lineRight

    def camera2body(self, img, sensor_calibration):
        ''' Tranform pixel coordinates to the vehicle body frame '''
        lanes_on_vehicle_coords = []

        yMax = img.shape[0]
        ploty = np.linspace(0, yMax - 1, yMax) * self.ym_per_pix
        lineLeft, lineRight = self.get_left_right_lane(ploty)

        intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])
        for lane in [lineLeft, lineRight]:
            lane_on_vehicle_coords = []
            for x, y in zip(lane, ploty):
                # camera_coords = np.dot(np.linalg.inv(intrinsic_matrix), np.array([l_u, l_v, 1]).reshape(3, 1))
                camera_coords = np.array([y, x, 1]).reshape(3, 1)
                # Transform camera to body frame.
                points = np.dot(np.linalg.inv(Quaternion(sensor_calibration['rotation']).rotation_matrix.T),
                                camera_coords)
                points = points + np.array(sensor_calibration['translation']).reshape((-1, 1))
                lane_on_vehicle_coords.append(points)
            lanes_on_vehicle_coords.append(lane_on_vehicle_coords)
        utils.plot_lanes(lanes_on_vehicle_coords)
        print('ok')
        return lanes_on_vehicle_coords

    def display_offset(self, img, input, fontScale=2):
        # unclear how it is defined
        yRange = 719

        # Calculate curvature
        leftCurvature = self.calculateCurvature(yRange, self.left_fit_m)
        rightCurvature = self.calculateCurvature(yRange, self.right_fit_m)

        # Calculate vehicle center
        xMax = img.shape[1] * self.xm_per_pix
        yMax = img.shape[0] * self.ym_per_pix
        vehicleCenter = xMax / 2
        lineLeft, lineRight = self.get_left_right_lane(yMax)
        lineMiddle = lineLeft + (lineRight - lineLeft) / 2
        diffFromVehicle = lineMiddle - vehicleCenter
        if diffFromVehicle > 0:
            message = '{:.2f} m right'.format(diffFromVehicle)
        else:
            message = '{:.2f} m left'.format(-diffFromVehicle)

        # Draw info
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (255, 255, 255)
        cv2.putText(input, 'Left curvature: {:.0f} m'.format(leftCurvature), (50, 50), font, fontScale, fontColor, 2)
        cv2.putText(input, 'Right curvature: {:.0f} m'.format(rightCurvature), (50, 120), font, fontScale, fontColor, 2)
        cv2.putText(input, 'Vehicle is {} of center'.format(message), (50, 190), font, fontScale, fontColor, 2)

        return input


class NuSceneProcessing:
    def __init__(self, camera_name, scene_id, root_path):
        self.camera_name = camera_name
        self.scene_id = scene_id

        self.nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
        self.nusc_map = NuScenesMap(dataroot=root_path, map_name='boston-seaport')

    def GetFilenameList(self):
        # Get the first sample in the selected scene
        current_sample = self.nusc.get('sample', self.nusc.scene[scene_id]['first_sample_token'])

        # Extract the filenames for the samples in the scene
        filename_list = []
        while not current_sample["next"] == "":
            sample_file_list = []
            for sensor in self.camera_name:
                sensor_data = self.nusc.get('sample_data', current_sample['data'][sensor])
                sample_file_list.append([sensor_data["filename"], current_sample["token"]])

            filename_list.append(sample_file_list)

            # Update the current sample with the next sample
            current_sample = self.nusc.get('sample', current_sample["next"])
        return filename_list

    def GetSensorCalibration(self):
        # Get the first sample in the selected scene
        current_sample = self.nusc.get('sample', self.nusc.scene[scene_id]['first_sample_token'])

        # Get the calibration data for each of the sensors
        calibration_data = []
        for sensor in self.camera_name:
            sensor_data = self.nusc.get('sample_data', current_sample['data'][sensor])
            calibration_data.append(self.nusc.get("calibrated_sensor", sensor_data["calibrated_sensor_token"]))

        return calibration_data


def main(sample_filenames, sensor_calibration_data):
    '''The main process to handel the image data'''
    for sample_filename in sample_filenames:
        for sensor_filename, sensor_calibration in zip(sample_filename, sensor_calibration_data):
            image = cv2.imread(root_path + sensor_filename[0])

            intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])
            undistort_img = cv2.undistort(image, intrinsic_matrix, None)
            hls_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2HLS)
            useSChannel = hls_img[:, :, 2]
            SobelX = lanedetection.absSobelThresh(useSChannel, thresh_min=10, thresh_max=160)
            SobelY = lanedetection.absSobelThresh(useSChannel, orient='y', thresh_min=10, thresh_max=160)

            resultCombined = lanedetection.combineGradients(SobelX, SobelY)

            img_unwarp = lanedetection.perspective_transform(resultCombined)
            left_fit, right_fit, _, _, _, _, _ = lanedetection.findLines(img_unwarp)
            new_img = lanedetection.drawLine(image, left_fit, right_fit)
            lanedetection.camera2body(image, sensor_calibration)
            input_img = lanedetection.display_offset(image, new_img)
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

    sample_filenames = data.GetFilenameList()
    sensor_calibration_data = data.GetSensorCalibration()

    main(sample_filenames, sensor_calibration_data)