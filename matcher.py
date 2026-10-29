import math
import cv2
import numpy as np


# ============================================================
# FEATURE WEIGHTS
# ============================================================

FEATURE_WEIGHTS = {

    "Aspect Ratio": 0.18,

    "Circularity": 0.18,

    "Solidity": 0.15,

    "Extent": 0.12,

    "Convexity": 0.12,

    "Elongation": 0.13,

    "Eccentricity": 0.12
}


# ============================================================
# MIRROR FIT THRESHOLDS
# ============================================================

MIRROR_HIGH_THRESHOLD = 85.0

MIRROR_POSSIBLE_THRESHOLD = 70.0


# ============================================================
# FEATURE SIMILARITY
# ============================================================

def feature_similarity(actual, expected):

    try:

        actual = float(actual)
        expected = float(expected)

    except (TypeError, ValueError):

        return 0.0

    if not math.isfinite(actual):
        return 0.0

    if not math.isfinite(expected):
        return 0.0

    denominator = abs(expected)

    if denominator < 1e-9:
        denominator = 1.0

    difference = abs(
        actual - expected
    )

    relative_error = (
        difference / denominator
    )

    similarity = (
        1.0 - relative_error
    )

    return max(
        0.0,
        min(
            1.0,
            similarity
        )
    )


# ============================================================
# REGION SCORE
# ============================================================

def calculate_region_score(
    shard_features,
    region_profile
):

    total_score = 0.0
    total_weight = 0.0

    feature_scores = {}
    matched_features = []

    for feature, weight in FEATURE_WEIGHTS.items():

        if feature not in shard_features:
            continue

        if feature not in region_profile:
            continue

        actual = shard_features[feature]

        expected = region_profile[feature]

        similarity = feature_similarity(
            actual,
            expected
        )

        feature_score = (
            similarity * 100.0
        )

        feature_scores[feature] = (
            feature_score
        )

        matched_features.append(
            feature
        )

        total_score += (
            similarity * weight
        )

        total_weight += weight

    if total_weight == 0:

        return {
            "score": 0.0,
            "feature_scores": {},
            "matched_features": [],
            "coverage": 0.0
        }

    score = (
        total_score /
        total_weight
    ) * 100.0

    coverage = (
        total_weight /
        sum(FEATURE_WEIGHTS.values())
    )

    return {

        "score": score,

        "feature_scores":
            feature_scores,

        "matched_features":
            matched_features,

        "coverage":
            coverage
    }


# ============================================================
# VESSEL COMPATIBILITY
# ============================================================

def calculate_vessel_compatibility(
    shard_features,
    vessel_profile
):

    region_results = {}

    for (
        region_name,
        region_profile
    ) in vessel_profile["regions"].items():

        result = calculate_region_score(
            shard_features,
            region_profile
        )

        region_results[
            region_name
        ] = result

    if not region_results:

        return {

            "best_region":
                None,

            "best_score":
                0.0,

            "regions":
                {}
        }

    best_region = max(
        region_results,
        key=lambda region:
            region_results[
                region
            ]["score"]
    )

    best_score = (
        region_results[
            best_region
        ]["score"]
    )

    return {

        "best_region":
            best_region,

        "best_score":
            best_score,

        "regions":
            region_results
    }


# ============================================================
# COMPATIBILITY LEVEL
# ============================================================

def get_compatibility_level(score):

    score = float(score)

    if score >= 80:

        return "High"

    elif score >= 65:

        return "Moderate"

    elif score >= 50:

        return "Low"

    else:

        return "Very Low"


# ============================================================
# FIND BEST VESSEL
# ============================================================

def find_best_vessel(
    shard_features,
    vessel_profiles
):

    vessel_results = []

    for (
        vessel_name,
        vessel_profile
    ) in vessel_profiles.items():

        compatibility = (
            calculate_vessel_compatibility(
                shard_features,
                vessel_profile
            )
        )

        score = (
            compatibility[
                "best_score"
            ]
        )

        region = (
            compatibility[
                "best_region"
            ]
        )

        level = (
            get_compatibility_level(
                score
            )
        )

        vessel_results.append({

            "vessel":
                vessel_name,

            "score":
                score,

            "region":
                region,

            "compatibility_level":
                level,

            "regions":
                compatibility[
                    "regions"
                ],

            "description":
                vessel_profile.get(
                    "description",
                    ""
                ),

            "reference_type":
                vessel_profile.get(
                    "reference_type",
                    "Prototype"
                )
        })

    vessel_results.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )

    return vessel_results


# ============================================================
# CONTOUR NORMALIZATION
# ============================================================

def normalize_contour(
    contour,
    num_points=200
):
    """
    Normalize a contour so that translation and scale
    do not affect the comparison.
    """

    if contour is None:
        return None

    points = contour.reshape(
        -1,
        2
    ).astype(
        np.float32
    )

    if len(points) < 3:
        return None

    # --------------------------------------------------------
    # Center at centroid
    # --------------------------------------------------------

    centroid = np.mean(
        points,
        axis=0
    )

    points = (
        points -
        centroid
    )

    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(
        distances
    )

    if scale <= 1e-9:
        return None

    points = (
        points /
        scale
    )

    # --------------------------------------------------------
    # Resample contour
    # --------------------------------------------------------

    points = resample_contour(
        points,
        num_points
    )

    return points


# ============================================================
# CONTOUR RESAMPLING
# ============================================================

def resample_contour(
    points,
    num_points=200
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) < 2:
        return points

    # Close contour
    closed = np.vstack([
        points,
        points[0]
    ])

    segment_lengths = np.linalg.norm(
        np.diff(
            closed,
            axis=0
        ),
        axis=1
    )

    cumulative = np.concatenate([
        [0.0],
        np.cumsum(
            segment_lengths
        )
    ])

    total_length = (
        cumulative[-1]
    )

    if total_length <= 1e-9:
        return points

    target_distances = np.linspace(
        0,
        total_length,
        num_points,
        endpoint=False
    )

    new_points = []

    for distance in target_distances:

        index = np.searchsorted(
            cumulative,
            distance,
            side="right"
        ) - 1

        index = max(
            0,
            min(
                index,
                len(points) - 1
            )
        )

        next_index = (
            index + 1
        ) % len(points)

        segment_start = (
            cumulative[index]
        )

        segment_end = (
            cumulative[index + 1]
        )

        segment_length = (
            segment_end -
            segment_start
        )

        if segment_length <= 1e-9:

            alpha = 0.0

        else:

            alpha = (
                distance -
                segment_start
            ) / segment_length

        point = (
            closed[index]
            +
            alpha *
            (
                closed[next_index]
                -
                closed[index]
            )
        )

        new_points.append(
            point
        )

    return np.asarray(
        new_points,
        dtype=np.float32
    )


# ============================================================
# CONTOUR DISTANCE
# ============================================================

def calculate_contour_distance(
    contour_a,
    contour_b
):
    """
    Calculate normalized mean point-to-point
    contour distance.
    """

    a = normalize_contour(
        contour_a
    )

    b = normalize_contour(
        contour_b
    )

    if a is None or b is None:

        return None

    if len(a) != len(b):

        return None

    distances = np.linalg.norm(
        a - b,
        axis=1
    )

    return float(
        np.mean(
            distances
        )
    )


# ============================================================
# CONTOUR SIMILARITY
# ============================================================

def contour_similarity(
    contour_a,
    contour_b
):
    """
    Convert contour distance into a percentage.

    Smaller contour distance = higher similarity.
    """

    distance = calculate_contour_distance(
        contour_a,
        contour_b
    )

    if distance is None:

        return 0.0

    # Empirical prototype conversion.
    #
    # A normalized distance of 0.0 means identical.
    # Larger distance reduces the score.

    similarity = (
        1.0 -
        distance
    )

    similarity = max(
        0.0,
        min(
            1.0,
            similarity
        )
    )

    return similarity * 100.0


# ============================================================
# MIRROR COMPATIBILITY
# ============================================================

def calculate_mirror_compatibility(
    mirrored_contour,
    opposite_reference_contour
):
    """
    Determine how closely the mirrored sherd contour
    matches an opposite-side reference contour.
    """

    if mirrored_contour is None:

        return {

            "available":
                False,

            "score":
                0.0,

            "decision":
                "No sherd contour",

            "distance":
                None
        }

    if opposite_reference_contour is None:

        return {

            "available":
                False,

            "score":
                0.0,

            "decision":
                "Reference contour required",

            "distance":
                None
        }

    distance = (
        calculate_contour_distance(
            mirrored_contour,
            opposite_reference_contour
        )
    )

    if distance is None:

        return {

            "available":
                False,

            "score":
                0.0,

            "decision":
                "Contour comparison failed",

            "distance":
                None
        }

    score = (
        contour_similarity(
            mirrored_contour,
            opposite_reference_contour
        )
    )

    if score >= MIRROR_HIGH_THRESHOLD:

        decision = (
            "Strong possible opposite-side match"
        )

    elif score >= MIRROR_POSSIBLE_THRESHOLD:

        decision = (
            "Possible opposite-side match"
        )

    else:

        decision = (
            "Unlikely opposite-side match"
        )

    return {

        "available":
            True,

        "score":
            score,

        "distance":
            distance,

        "decision":
            decision
    }


# ============================================================
# COMPARE ORIGINAL AND MIRROR
# ============================================================

def compare_match_results(
    original_result,
    mirror_result
):

    original_score = float(
        original_result["score"]
    )

    mirror_score = float(
        mirror_result["score"]
    )

    difference = abs(
        original_score -
        mirror_score
    )

    same_vessel = (
        original_result["vessel"]
        ==
        mirror_result["vessel"]
    )

    same_region = (
        original_result["region"]
        ==
        mirror_result["region"]
    )

    same_score = (
        difference < 0.01
    )

    return {

        "same_vessel":
            same_vessel,

        "same_region":
            same_region,

        "same_score":
            same_score,

        "score_difference":
            difference,

        "mirror_shape_equivalent":
            (
                same_vessel
                and
                same_region
                and
                same_score
            )
    }
