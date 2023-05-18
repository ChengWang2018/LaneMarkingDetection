import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from PIL import Image, ImageDraw
from math import sqrt


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


def plot_lane_on_map(cam_path, lanes_on_vehicle_coords, gt):
    ''' Plot detected left and right lanes on the ground truth (map) and also plot the lane lines of the map'''
    ''' Plot on image coordinate system'''
    im = Image.open(cam_path)
    im_size = im.size

    fig, ax = plt.subplots(1, 1)
    color = ['#D49AB3', '#7A93B2']

    for lane_on_vehicle_coords in lanes_on_vehicle_coords:
        coord_x = []
        coord_y = []
        lane_points = []
        for coord in lane_on_vehicle_coords:
            coord_x.append(coord[0])
            coord_y.append(coord[1])
            lane_points.append((coord[0], coord[1]))
        ax.plot(coord_x, coord_y, color='b')

        # plot line on image
        img1 = ImageDraw.Draw(im)
        img1.line(lane_points, fill='#FF0000', width=6)

    # plot the ground truth
    gt_x = []
    gt_y = []
    gt_temp = []
    for p in gt:
        if p[0] > 0 and p[1] > 0:
            gt_x.append(p[0])
            gt_y.append(p[1])
            gt_temp.append((p[0], p[1]))
    ax.scatter(gt_x, gt_y, color='r')
    plt.xlim([0, im_size[0]])
    plt.ylim([0, im_size[1]])
    plt.gca().invert_yaxis()

    # plot gt on image
    img1 = ImageDraw.Draw(im)
    img1.line(gt_temp, fill='#0000FF', width=6)
    im.show()

    im.save('detected_results.png')
    plt.show()


def plot_gt(gt, middle_line, resampled_gt):
    '''plot the ground truth lane points and also the resampled ground truth'''
    fig, ax = plt.subplots(1, 1)
    # plot the ground truth
    gt_x = []
    gt_y = []
    for p in gt:
        gt_x.append(p[0])
        gt_y.append(p[1])
    ax.scatter(gt_x, gt_y, color='b')

    # plot the middle line
    for p in middle_line:
        ax.scatter(p[0], p[1], color='y')

    # plot the resampled ground truth
    for p in resampled_gt:
        ax.scatter(p[0], p[1], color='r')

    plt.show()


def plot_on_perspective(detected_lines, ego_pos, gt_persepctive):
    '''plot detected lines and ground truth on the perspective image'''
    fig, ax = plt.subplots(1, 1)

    for detected_line in detected_lines[1:]:
        if detected_line is not None:
            ax.scatter(detected_line, detected_lines[0], color='b')
    # plot ego pos on the perspective image
    ax.scatter(ego_pos[0], ego_pos[1], color='y', label='Ego')

    # plot ground truth on perspective image
    gt_x = []
    gt_y = []
    for gt in gt_persepctive:
        gt_x.append(gt[0])
        gt_y.append(gt[1])
    ax.scatter(gt_x, gt_y, color='r')

    plt.gca().invert_yaxis()
    plt.show()


def get_euclidean_distance(point1, point2):
    dist_square = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    return sqrt(float(dist_square))


def solve_equation(alpha, curr_point, next_point, dis):
    ''' get a point with defined distance to the beginning point'''

    x = Symbol('x')
    y = Symbol('y')
    solved_values = solve(
        [(curr_point[1] - y) - alpha * (curr_point[0] - x),
         (curr_point[1] - y) ** 2 + (curr_point[0] - x) ** 2 - dis ** 2], [x, y])

    # get the value that is closer to the next point
    dis_temp = 9999
    solved = [None, None]
    for value in solved_values:
        dis_ = get_euclidean_distance(value, next_point)
        if dis_ < dis_temp:
            dis_temp = dis_
            solved[0] = float(value[0])
            solved[1] = float(value[1])
    return solved
