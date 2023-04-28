import numpy as np
import matplotlib.pyplot as plt
from sympy import *


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


def plot_lane_on_map(lanes_on_vehicle_coords, gt):
    ''' Plot left and right lanes on the ground truth (map)'''
    fig, ax = plt.subplots(1, 1)
    for lane_on_vehicle_coords in lanes_on_vehicle_coords:
        coord_x = []
        coord_y = []
        for coord in lane_on_vehicle_coords:
            coord_x.append(coord[0])
            coord_y.append(coord[1])
        ax.plot(coord_x, coord_y, color='b')

    # plot the ground truth
    gt_x = []
    gt_y = []
    for p in gt:
        gt_x.append(p[0])
        gt_y.append(p[1])
    ax.scatter(gt_x, gt_y, color='r')
    plt.gca().invert_yaxis()
    plt.show()

def plot_gt(gt, resampled_gt):
    '''plot the ground truth lane points'''
    fig, ax = plt.subplots(1, 1)
    # plot the ground truth
    gt_x = []
    gt_y = []
    for p in gt:
        gt_x.append(p[0])
        gt_y.append(p[1])
    ax.scatter(gt_x, gt_y, color='b')

    # plot the resampled ground truth
    r_gt_x = []
    r_gt_y = []
    for p in resampled_gt:
        ax.scatter(p[0], p[1], color='r')

    plt.show()


def get_euclidean_distance(point1, point2):
    dist_square = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    return np.sqrt(float(dist_square))


def solve_equation(alpha, curr_point, next_point, dis):
    ''' get a point with defined distance to the beginning point'''

    x = Symbol('x')
    y = Symbol('y')
    solved_values = solve(
        [(curr_point[1] - y) - alpha * (curr_point[0] - x),
         (curr_point[1] - y) ** 2 + (curr_point[0] - x) ** 2 - dis ** 2], [x, y])

    # get the value that is closer to the next point
    dis = 9999
    solved = [None, None]
    for value in solved_values:
        dis_ = get_euclidean_distance(value, next_point)
        if dis_ < dis:
            dis = dis_
            solved[0] = float(value[0])
            solved[1] = float(value[1])
    return solved
