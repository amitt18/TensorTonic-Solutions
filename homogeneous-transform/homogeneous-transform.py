import numpy as np

def apply_homogeneous_transform(T, points):

    points = np.asarray(points)

    # Check if input is a single point
    single = points.ndim == 1
    if single:
        points = points[np.newaxis, :]

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Apply transformation
    transformed = (T @ points_h.T).T

    # Remove homogeneous coordinate
    result = transformed[:, :3]

    return result[0] if single else result