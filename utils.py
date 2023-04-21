
import numpy as np
import matplotlib.pyplot as plt

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
