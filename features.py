import cv2
import numpy as np
import math


def calculate_features(contour):
    """
    Calculate mathematical/geometrical features
    from the detected pottery sherd contour.
    """

    # ---------------------------------------------------------
    # BASIC MEASUREMENTS
    # ---------------------------------------------------------

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 0 or perimeter <= 0:
        raise ValueError("Invalid contour detected.")

    x, y, width, height = cv2.boundingRect(contour)

    bounding_box_area = float(width * height)

    # Aspect ratio
    aspect_ratio = width / height if height > 0 else 0.0

    # Extent
    extent = area / bounding_box_area if bounding_box_area > 0 else 0.0

    # Circularity
    circularity = (
        (4.0 * math.pi * area) / (perimeter * perimeter)
        if perimeter > 0 else 0.0
    )

    # Compactness
    compactness = (
        (perimeter * perimeter) / area
        if area > 0 else 0.0
    )

    # Form factor
    form_factor = circularity

    # Equivalent diameter
    equivalent_diameter = math.sqrt(
        (4.0 * area) / math.pi
    )

    # ---------------------------------------------------------
    # CONVEX HULL
    # ---------------------------------------------------------

    hull = cv2.convexHull(contour)

    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    solidity = (
        area / hull_area
        if hull_area > 0 else 0.0
    )

    convex_deficiency = hull_area - area

    convexity = (
        hull_perimeter / perimeter
        if perimeter > 0 else 0.0
    )

    hull_area_ratio = (
        hull_area / area
        if area > 0 else 0.0
    )

    # ---------------------------------------------------------
    # CENTROID AND MOMENTS
    # ---------------------------------------------------------

    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

    else:

        centroid_x = 0.0
        centroid_y = 0.0

    # ---------------------------------------------------------
    # ROTATED RECTANGLE
    # ---------------------------------------------------------

    rotated_rect = cv2.minAreaRect(contour)

    (center_x, center_y), (rect_width, rect_height), angle = rotated_rect

    rotated_area = rect_width * rect_height

    rectangularity = (
        area / rotated_area
        if rotated_area > 0 else 0.0
    )

    # Major/minor axis
    major_axis = max(rect_width, rect_height)
    minor_axis = min(rect_width, rect_height)

    elongation = (
        major_axis / minor_axis
        if minor_axis > 0 else 0.0
    )

    eccentricity = math.sqrt(
        max(
            0.0,
            1.0 - (
                (minor_axis * minor_axis) /
                (major_axis * major_axis)
            )
        )
    ) if major_axis > 0 else 0.0

    # ---------------------------------------------------------
    # FERET DIAMETER
    # ---------------------------------------------------------

    points = contour.reshape(-1, 2).astype(np.float32)

    # Limit points for speed
    if len(points) > 300:
        indices = np.linspace(
            0,
            len(points) - 1,
            300
        ).astype(int)

        points = points[indices]

    max_feret = 0.0

    for i in range(len(points)):

        distances = np.linalg.norm(
            points[i + 1:] - points[i],
            axis=1
        )

        if len(distances) > 0:
            max_feret = max(
                max_feret,
                float(np.max(distances))
            )

    min_feret = min(
        rect_width,
        rect_height
    )

    # ---------------------------------------------------------
    # HU MOMENTS
    # ---------------------------------------------------------

    hu = cv2.HuMoments(moments).flatten()

    hu_features = []

    for value in hu:

        if value == 0:
            hu_features.append(0.0)

        else:
            hu_features.append(
                -np.sign(value) *
                np.log10(abs(value))
            )

    # ---------------------------------------------------------
    # RETURN ALL FEATURES
    # ---------------------------------------------------------

    return {

        # Basic
        "Area (px²)": area,
        "Perimeter (px)": perimeter,
        "Bounding Box X (px)": x,
        "Bounding Box Y (px)": y,
        "Width (px)": width,
        "Height (px)": height,
        "Bounding Box Area (px²)": bounding_box_area,

        # Shape
        "Aspect Ratio": aspect_ratio,
        "Extent": extent,
        "Circularity": circularity,
        "Compactness": compactness,
        "Form Factor": form_factor,
        "Equivalent Diameter (px)": equivalent_diameter,

        # Convexity
        "Convex Hull Area (px²)": hull_area,
        "Convex Hull Perimeter (px)": hull_perimeter,
        "Solidity": solidity,
        "Convex Deficiency (px²)": convex_deficiency,
        "Convexity": convexity,
        "Hull Area Ratio": hull_area_ratio,

        # Centroid
        "Centroid X (px)": centroid_x,
        "Centroid Y (px)": centroid_y,

        # Raw moments
        "Moment M00": moments["m00"],
        "Moment M10": moments["m10"],
        "Moment M01": moments["m01"],
        "Moment M20": moments["m20"],
        "Moment M11": moments["m11"],
        "Moment M02": moments["m02"],

        # Central moments
        "Central Mu20": moments["mu20"],
        "Central Mu11": moments["mu11"],
        "Central Mu02": moments["mu02"],
        "Central Mu30": moments["mu30"],
        "Central Mu21": moments["mu21"],
        "Central Mu12": moments["mu12"],
        "Central Mu03": moments["mu03"],

        # Normalized moments
        "Normalized Nu20": moments["nu20"],
        "Normalized Nu11": moments["nu11"],
        "Normalized Nu02": moments["nu02"],
        "Normalized Nu30": moments["nu30"],
        "Normalized Nu21": moments["nu21"],
        "Normalized Nu12": moments["nu12"],
        "Normalized Nu03": moments["nu03"],

        # Rotated rectangle
        "Rotated Rectangle Width (px)": rect_width,
        "Rotated Rectangle Height (px)": rect_height,
        "Rotated Rectangle Area (px²)": rotated_area,
        "Orientation (degrees)": angle,

        "Rectangularity": rectangularity,

        # Axis features
        "Major Axis Length (px)": major_axis,
        "Minor Axis Length (px)": minor_axis,
        "Elongation": elongation,
        "Eccentricity": eccentricity,

        # Feret
        "Maximum Feret Diameter (px)": max_feret,
        "Minimum Feret Diameter (px)": min_feret,

        # Ratios
        "Perimeter / Area Ratio":
            perimeter / area,

        "Area / Perimeter Ratio":
            area / perimeter,

        "Perimeter / Hull Perimeter":
            perimeter / hull_perimeter
            if hull_perimeter > 0 else 0.0,

        # Hu moments
        "Hu Moment 1": hu_features[0],
        "Hu Moment 2": hu_features[1],
        "Hu Moment 3": hu_features[2],
        "Hu Moment 4": hu_features[3],
        "Hu Moment 5": hu_features[4],
        "Hu Moment 6": hu_features[5],
        "Hu Moment 7": hu_features[6]
    }
