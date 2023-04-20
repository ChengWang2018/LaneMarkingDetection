import numpy as np


def clip_points_behind_camera(points, near_plane: float):
    """
    Perform clipping on polygons that are partially behind the camera.
    This method is necessary as the projection does not work for points behind the camera.
    Hence we compute the line between the point and the camera and follow that line until we hit the near plane of
    the camera. Then we use that point.
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param near_plane: If we set the near_plane distance of the camera to 0 then some points will project to
        infinity. Therefore we need to clip these points at the near plane.
    :return: The clipped version of the polygon. This may have fewer points than the original polygon if some lines
        were entirely behind the polygon.
    """
    points_clipped = []
    # Loop through each line on the polygon.
    # For each line where exactly 1 endpoints is behind the camera, move the point along the line until
    # it hits the near plane of the camera (clipping).
    assert points.shape[0] == 3
    point_count = points.shape[1]
    for line_1 in range(point_count):
        line_2 = (line_1 + 1) % point_count
        point_1 = points[:, line_1]
        point_2 = points[:, line_2]
        z_1 = point_1[2]
        z_2 = point_2[2]

        if z_1 >= near_plane and z_2 >= near_plane:
            # Both points are in front.
            # Add both points unless the first is already added.
            if len(points_clipped) == 0 or all(points_clipped[-1] != point_1):
                points_clipped.append(point_1)
            points_clipped.append(point_2)
        elif z_1 < near_plane and z_2 < near_plane:
            # Both points are in behind.
            # Don't add anything.
            continue
        else:
            # One point is in front, one behind.
            # By convention pointA is behind the camera and pointB in front.
            if z_1 <= z_2:
                point_a = points[:, line_1]
                point_b = points[:, line_2]
            else:
                point_a = points[:, line_2]
                point_b = points[:, line_1]
            z_a = point_a[2]
            z_b = point_b[2]

            # Clip line along near plane.
            pointdiff = point_b - point_a
            alpha = (near_plane - z_b) / (z_a - z_b)
            clipped = point_a + (1 - alpha) * pointdiff
            assert np.abs(clipped[2] - near_plane) < 1e-6

            # Add the first point (if valid and not duplicate), the clipped point and the second point (if valid).
            if z_1 >= near_plane and (len(points_clipped) == 0 or all(points_clipped[-1] != point_1)):
                points_clipped.append(point_1)
            points_clipped.append(clipped)
            if z_2 >= near_plane:
                points_clipped.append(point_2)

    points_clipped = np.array(points_clipped).transpose()
    return points_clipped