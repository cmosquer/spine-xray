def findLowestPoint(keypoints):
    """ Finds lowest point from a list of keypoints

    Args:
        keypoints (numpy.array(int)): array that contains all keypoints values 

    Returns:
        int: lowest keypoint (y coordinate)
    """
    # keypoins notation [x,y]
    max_y = keypoints[0][1]
    for _, y in keypoints:
        if y > max_y: max_y = y
    return int(max_y)


def findHighestPoint(keypoints):
    """ Finds highest point from a list of keypoints

    Args:
        keypoints (numpy.array(int)): array that contains all keypoints values 

    Returns:
        int: Highest keypoint (y coordinate)
    """
    # keypoins notation [x,y]
    min_y = keypoints[0][1]  # Initialized min with first point
    for _, y in keypoints:
        if y < min_y: min_y = y
    return int(min_y)


def findFarLeftPoint(keypoints):
    """ Finds farthest left point from a list of keypoints

    Args:
        keypoints (numpy.array(int)): array that contains all keypoints values 

    Returns:
        int: Fardest left keypoint (y coordinate)
    """
    # keypoins notation [x,y]
    min_x = keypoints[0][0]  # Initialized min with first point
    for x, _ in keypoints:
        if x < min_x: min_x = x
    return int(min_x)


def findFarRightPoint(keypoints):
    """ Finds farthest right point from a list of keypoints

    Args:
        keypoints (numpy.array(int)): array that contains all keypoints values 

    Returns:
        int: Fardest right keypoint (y coordinate)
    """
    # keypoins notation [x,y]
    max_x = keypoints[0][0]  # Initialized min with first point
    for x, _ in keypoints:
        if x > max_x: max_x = x
    return int(max_x)


def getBoundingBox(keypoints, expand_px = 30):
    """ Returns two vertices of a bounding box created using the list of keypoints send as a parameter 

    Args:
        keypoints (numpy.array(int)): array that contains all keypoints values 
        expand_px (int, optional): pixel used to expand the bounding box created. Defaults to 30.

    Returns:
        int, int, int, int: x0 and y0 associated to top left vertice follow by x1 and y1 associated to bottom right vertice
    """
    y0, x0 = findHighestPoint(keypoints) - expand_px, findFarLeftPoint(keypoints) - expand_px
    y1, x1 = findLowestPoint(keypoints) + expand_px, findFarRightPoint(keypoints) + expand_px
    if y0 < 0: y0 = 0
    if x0 < 0: x0 = 0
    if y1 < 0: y1 = 0
    if x1 < 0: x1 = 0
    return x0, y0, x1, y1
