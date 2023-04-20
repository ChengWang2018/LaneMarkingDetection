from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import matplotlib.pyplot as plt
import numpy as np
import cv2

ym_per_pix = 0.2873
xm_per_pix = 0.02555


def TransformImagePerspective(source_points, input_image):
    bottom_width = source_points[2, 0] - source_points[3, 0]
    height = source_points[2, 1] - source_points[0, 1]

    destination_points = np.array([
        [0, 0],
        [bottom_width - 1, 0],
        [bottom_width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    Minv = cv2.getPerspectiveTransform(destination_points, source_points)

    transformed_image = cv2.warpPerspective(input_image, M, (int(bottom_width), int(height)))

    return transformed_image, Minv, M


def GetFilenameList(sensor_list=['CAM_FRONT'], scene_id=0):
    # Get the first sample in the selected scene
    current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])

    # Extract the filenames for the samples in the scene
    filename_list = []
    while not current_sample["next"] == "":
        sample_file_list = []
        for sensor in sensor_list:
            sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
            sample_file_list.append([sensor_data["filename"], current_sample["token"]])

        filename_list.append(sample_file_list)

        # Update the current sample with the next sample
        current_sample = nusc.get('sample', current_sample["next"])
    return filename_list


def GetSensorCalibration(sensor_list=['CAM_FRONT'], scene_id=0):
    # Get the first sample in the selected scene
    current_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])

    # Get the calibration data for each of the sensors
    calibration_data = []
    for sensor in sensor_list:
        sensor_data = nusc.get('sample_data', current_sample['data'][sensor])
        calibration_data.append(nusc.get("calibrated_sensor", sensor_data["calibrated_sensor_token"]))

    return calibration_data


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


def findLines(binary_warped, nwindows=9, margin=110, minpix=50):
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
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
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
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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
    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)


def visualizeLanes(image, ax):
    """
    Visualize the windows and fitted lines for `image`.
    Returns (`left_fit` and `right_fit`)
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = findLines(
        image)
    # Visualization
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax.imshow(out_img)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    return (left_fit, right_fit, left_fit_m, right_fit_m)


def showLaneOnImages(img, cols=1, rows=1, figsize=(15, 13)):
    """
    Display `images` on a [`cols`, `rows`] subplot grid.
    Returns a collection with the image paths and the left and right polynomials.
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    left_fit, right_fit, left_fit_m, right_fit_m = visualizeLanes(img, ax)
    ax.set_title('LanePoly')
    ax.axis('off')
    imageAndFit = (left_fit, right_fit, left_fit_m, right_fit_m)
    return imageAndFit


def showImages(img, cols=1, rows=1, figsize=(15, 10), cmap=None):
    """
    Display `images` on a [`cols`, `rows`] subplot grid.
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    if cmap == None:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap)
    ax.set_title('gradients')
    ax.axis('off')


def threshIt(img, thresh_min, thresh_max):
    """
    Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
    """
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return xbinary


def absSobelThresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
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
    return threshIt(scaled, thresh_min, thresh_max)


def combineGradients(sobelX, sobelY):
    """
    Compute the combination of Sobel X and Sobel Y or Magnitude and Direction
    """
    combined = np.zeros_like(sobelX)
    combined[((sobelX == 1) & (sobelY == 1))] = 1
    return combined


def calculateCurvature(yRange, left_fit_cr):
    """
    Returns the curvature of the polynomial `fit` on the y range `yRange`.
    """

    return ((1 + (2 * left_fit_cr[0] * yRange * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])


def drawLine(img, left_fit, right_fit):
    """
    Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
    """
    yMax = img.shape[0]
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
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


if __name__ == '__main__':
    # sensor_names = ["CAM_FRONT","CAM_BACK"]
    sensor_names = ["CAM_FRONT"]
    scene_id = 3
    root_path = 'D:/Work/nuscene/data/sets/nuscenes/'
    nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
    nusc_map = NuScenesMap(dataroot=root_path, map_name='boston-seaport')

    lower_limit = np.array([120, 150, 180])
    upper_limit = np.array([220, 220, 220])
    kernel = np.ones((15, 15), np.uint8)

    # Point order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    source_points = np.array([[680, 565], [970, 565], [1350, 900], [190, 900]], dtype="float32")

    sensor_filenames = GetFilenameList(sensor_names, scene_id)
    sensor_calibration_data = GetSensorCalibration(sensor_names, scene_id)

    yRange = 719

    for sample_filenames in sensor_filenames:
        for sensor_filename, sensor_calibration in zip(sample_filenames, sensor_calibration_data):
            image = cv2.imread(root_path + sensor_filename[0])

            intrinsic_matrix = np.array(sensor_calibration["camera_intrinsic"])
            undistort_img = cv2.undistort(image, intrinsic_matrix, None)
            hls_img = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2HLS)
            useSChannel = hls_img[:, :, 2]
            SobelX = absSobelThresh(useSChannel, thresh_min=10, thresh_max=160)
            SobelY = absSobelThresh(useSChannel, orient='y', thresh_min=10, thresh_max=160)

            resultCombined = combineGradients(SobelX, SobelY)
            # showImages(resultCombined, 1, 1, (15, 13), cmap='gray')

            # img = draw_polygon_on_image(source_points, image)
            # cv2.imwrite("polygonOnimage.jpg", img)

            img_unwarp, Minv, M = TransformImagePerspective(source_points, resultCombined)
            left_fit, right_fit, left_fit_m, right_fit_m, _, _, _, _, _ = findLines(img_unwarp)
            output = drawLine(image, left_fit, right_fit)

            # Calculate curvature
            leftCurvature = calculateCurvature(yRange, left_fit_m)
            rightCurvature = calculateCurvature(yRange, right_fit_m)

            #
            # histogram = hist(vis)
            # plt.plot(histogram)
            plt.show()

    cv2.destroyAllWindows()
