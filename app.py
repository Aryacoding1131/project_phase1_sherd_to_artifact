import os
import math
import uuid
import hashlib
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from PIL import Image
from features import calculate_features
from vessel_profiles import VESSEL_PROFILES, PROFILE_ERRORS
from matcher import find_best_vessel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pottery Sherd Analysis",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FEATURE_WEIGHTS = {
    "Aspect Ratio": 0.18,
    "Circularity": 0.18,
    "Solidity": 0.15,
    "Extent": 0.12,
    "Convexity": 0.12,
    "Elongation": 0.13,
    "Eccentricity": 0.12,
}

# ============================================================
# VESSEL DIMENSIONS FOR 3D VISUALIZATION
# These are prototype visualization dimensions.
# They are not archaeological measurements.
# ============================================================

VESSEL_DIMENSIONS = {
    "A - Small Bowl": {
        "height": 4.0,
        "max_radius": 3.0,
        "regions": {
            "Rim": (3.5, 2.9),
            "Upper Body": (2.7, 2.7),
            "Lower Body": (1.5, 2.3),
            "Base": (0.4, 1.8),
        },
    },

    "B - Storage Pot": {
        "height": 6.0,
        "max_radius": 3.2,
        "regions": {
            "Rim": (5.7, 2.5),
            "Neck": (5.0, 2.0),
            "Shoulder": (4.0, 3.0),
            "Body": (2.5, 3.2),
            "Base": (0.5, 2.0),
        },
    },

    "C - Carinated Vase": {
        "height": 6.0,
        "max_radius": 2.8,
        "regions": {
            "Rim": (5.7, 2.0),
            "Neck": (5.0, 1.6),
            "Shoulder": (4.0, 2.8),
            "Body": (2.5, 2.4),
            "Base": (0.5, 1.8),
        },
    },

    "D - Corrugated Bowl": {
        "height": 4.5,
        "max_radius": 3.2,
        "regions": {
            "Rim": (4.2, 3.1),
            "Upper Body": (3.0, 2.9),
            "Lower Body": (1.5, 2.4),
            "Base": (0.4, 1.8),
        },
    },

    "E - Tall Vessel": {
        "height": 6.5,
        "max_radius": 2.5,
        "regions": {
            "Rim": (6.2, 1.9),
            "Neck": (5.5, 1.7),
            "Shoulder": (4.6, 2.3),
            "Body": (2.7, 2.5),
            "Base": (0.5, 1.6),
        },
    },
}


# ============================================================
# BACKGROUND REMOVAL AND CONTOUR DETECTION
# ============================================================

def remove_background(image):
    """
    Remove the background using OpenCV GrabCut.

    Returns:
        foreground_rgba: image with transparent background
        final_mask: binary foreground mask
        largest_contour: detected outer contour
    """

    try:
        rgb_image = np.array(image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        height, width = bgr_image.shape[:2]

        if height < 10 or width < 10:
            return None, None, None

        mask = np.zeros((height, width), dtype=np.uint8)

        border = max(
            1,
            min(5, height // 20, width // 20)
        )

        rect_width = max(1, width - 2 * border)
        rect_height = max(1, height - 2 * border)

        rect = (
            border,
            border,
            rect_width,
            rect_height
        )

        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(
            bgr_image,
            mask,
            rect,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT
        )

        foreground_mask = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        # Reduce small gaps and noise.
        kernel = np.ones((5, 5), np.uint8)

        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            foreground_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None, None

        # Ignore very small detected regions.
        min_area = height * width * 0.005

        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]

        if not valid_contours:
            return None, None, None

        largest_contour = max(
            valid_contours,
            key=cv2.contourArea
        )

        final_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        rgba_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2RGBA
        )

        rgba_image[:, :, 3] = final_mask

        foreground_rgba = Image.fromarray(rgba_image)

        return foreground_rgba, final_mask, largest_contour

    except Exception as error:
        st.error(f"Background removal failed: {error}")
        return None, None, None


# ============================================================
# CONTOUR DRAWING
# ============================================================

def draw_contour_on_image(image, contour):
    """Draw the detected contour on the original image."""

    image_array = np.array(image.convert("RGB"))

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    cv2.drawContours(
        image_bgr,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(image_rgb)


# ============================================================
# MIRROR ANALYSIS
# ============================================================

def mirror_contour(contour):
    """Mirror the contour horizontally around its bounding-box center."""

    mirrored = contour.copy()

    x, y, width, height = cv2.boundingRect(contour)

    center_x = x + width / 2.0

    mirrored[:, :, 0] = (
        2 * center_x - contour[:, :, 0]
    ).astype(np.int32)

    return mirrored


def create_mirrored_mask(mask):
    """Create a horizontally mirrored version of the mask."""

    return cv2.flip(mask, 1)


def calculate_mirror_symmetry(original_mask, mirrored_mask):
    """
    Calculate similarity between the original and mirrored masks.

    IoU measures intersection over union.
    Dice measures overlap between the two masks.
    """

    original_binary = original_mask > 0
    mirrored_binary = mirrored_mask > 0

    intersection = np.logical_and(
        original_binary,
        mirrored_binary
    ).sum()

    union = np.logical_or(
        original_binary,
        mirrored_binary
    ).sum()

    original_area = original_binary.sum()
    mirrored_area = mirrored_binary.sum()

    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    dice_denominator = original_area + mirrored_area

    if dice_denominator == 0:
        dice = 0.0
    else:
        dice = (
            2.0 * intersection / dice_denominator
        )

    score = ((iou + dice) / 2.0) * 100.0

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Score": float(score),
    }


# ============================================================
# FEATURE TABLE
# ============================================================

def create_feature_comparison(original_features, mirrored_features):
    """Create a comparison table for original and mirrored features."""

    rows = []

    all_keys = list(original_features.keys())

    for key in mirrored_features:
        if key not in all_keys:
            all_keys.append(key)

    for key in all_keys:
        original_value = original_features.get(key)
        mirrored_value = mirrored_features.get(key)

        difference = None

        try:
            if (
                original_value is not None
                and mirrored_value is not None
            ):
                difference = abs(
                    float(original_value)
                    - float(mirrored_value)
                )
        except (ValueError, TypeError):
            difference = None

        rows.append({
            "Feature": key,
            "Original Sherd": original_value,
            "Mirrored Sherd": mirrored_value,
            "Absolute Difference": difference,
        })

    return pd.DataFrame(rows)


# ============================================================
# VESSEL PROFILE HELPERS
# ============================================================

def get_vessel_dimensions(vessel_name):
    """Return the dimensions used to draw a vessel."""

    if vessel_name in VESSEL_DIMENSIONS:
        return VESSEL_DIMENSIONS[vessel_name]

    return {
        "height": 6.0,
        "max_radius": 2.5,
        "regions": {
            "Rim": (5.7, 2.0),
            "Neck": (4.8, 1.8),
            "Body": (2.5, 2.5),
            "Base": (0.5, 1.5),
        },
    }


def get_vessel_profile(vessel_name):
    """Generate a simplified parametric profile for a vessel."""

    dimensions = get_vessel_dimensions(vessel_name)

    height = dimensions["height"]
    max_radius = dimensions["max_radius"]

    z_factors = np.array([
        0.00,
        0.10,
        0.25,
        0.45,
        0.65,
        0.82,
        0.94,
        1.00,
    ])

    radius_factors = np.array([
        0.55,
        0.65,
        0.85,
        1.00,
        0.90,
        0.65,
        0.55,
        0.50,
    ])

    z_values = z_factors * height
    radius_values = radius_factors * max_radius

    return z_values, radius_values


def radius_at_height(z_value, profile):
    """Estimate the vessel radius at a particular height."""

    z_values, radius_values = profile

    return float(
        np.interp(
            z_value,
            z_values,
            radius_values
        )
    )


# ============================================================
# CREATE 3D VESSEL SURFACE
# ============================================================

def create_vessel_surface(vessel_name):
    """Create a parametric 3D vessel surface."""

    dimensions = get_vessel_dimensions(vessel_name)
    profile = get_vessel_profile(vessel_name)

    z_values, radius_values = profile

    theta = np.linspace(
        0,
        2 * np.pi,
        100
    )

    z_grid = np.linspace(
        0,
        dimensions["height"],
        100
    )

    theta_grid, height_grid = np.meshgrid(
        theta,
        z_grid
    )

    radius_grid = np.interp(
        height_grid,
        z_values,
        radius_values
    )

    x_grid = radius_grid * np.cos(theta_grid)
    y_grid = radius_grid * np.sin(theta_grid)
    z_grid_surface = height_grid

    return (
        x_grid,
        y_grid,
        z_grid_surface,
        dimensions,
        profile,
    )


# ============================================================
# CREATE SHERD MESH FOR 3D VISUALIZATION
# ============================================================

def create_3d_sherd(
    contour,
    vessel_name,
    region_name
):
    """
    Map the sherd silhouette onto the approximate surface
    of the selected region of a parametric vessel.
    """

    if contour is None:
        return None

    dimensions = get_vessel_dimensions(vessel_name)
    regions = dimensions["regions"]

    if region_name not in regions:
        return None

    region_z, region_radius = regions[region_name]

    x, y, width, height = cv2.boundingRect(contour)

    if width <= 0 or height <= 0:
        return None

    # Create a cropped binary silhouette.
    local_contour = contour.copy()

    local_contour[:, :, 0] -= x
    local_contour[:, :, 1] -= y

    silhouette = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.drawContours(
        silhouette,
        [local_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Scale the silhouette to the selected vessel region.
    max_width = min(
        region_radius * 0.85,
        2.2
    )

    max_height = dimensions["height"] * 0.22

    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1)
    )

    new_width = max(
        3,
        int(width * scale * 20)
    )

    new_height = max(
        3,
        int(height * scale * 20)
    )

    # Keep the mesh resolution manageable.
    new_width = min(new_width, 100)
    new_height = min(new_height, 100)

    resized_mask = cv2.resize(
        silhouette,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Center the sherd around the selected region.
    center_x = new_width / 2.0
    center_y = new_height / 2.0

    scale_x = max_width / new_width
    scale_z = max_height / new_height

    rows, cols = np.where(resized_mask > 0)

    if len(rows) < 3:
        return None

    # Place the sherd on the front of the vessel.
    x_coordinates = (
        cols - center_x
    ) * scale_x

    z_coordinates = (
        region_z
        - (rows - center_y) * scale_z
    )

    z_coordinates = np.clip(
        z_coordinates,
        0,
        dimensions["height"]
    )

    profile = get_vessel_profile(vessel_name)

    y_coordinates = []

    for x_coordinate, z_coordinate in zip(
        x_coordinates,
        z_coordinates
    ):
        radius = radius_at_height(
            z_coordinate,
            profile
        )

        x_limited = np.clip(
            x_coordinate,
            -radius * 0.95,
            radius * 0.95
        )

        # Front surface of the vessel.
        y_coordinate = -math.sqrt(
            max(
                radius**2 - x_limited**2,
                0
            )
        )

        y_coordinates.append(
            y_coordinate - 0.035
        )

    # Triangulate the silhouette using neighboring pixels.
    index_map = np.full(
        (new_height, new_width),
        -1,
        dtype=np.int32
    )

    index_map[rows, cols] = np.arange(len(rows))

    i_indices = []
    j_indices = []
    k_indices = []

    for row in range(new_height - 1):
        for col in range(new_width - 1):

            p1 = index_map[row, col]
            p2 = index_map[row, col + 1]
            p3 = index_map[row + 1, col]
            p4 = index_map[row + 1, col + 1]

            if (
                p1 >= 0
                and p2 >= 0
                and p3 >= 0
                and p4 >= 0
            ):
                i_indices.extend([p1, p2])
                j_indices.extend([p2, p4])
                k_indices.extend([p3, p3])

    if not i_indices:
        return None

    return go.Mesh3d(
        x=x_coordinates,
        y=y_coordinates,
        z=z_coordinates,
        i=i_indices,
        j=j_indices,
        k=k_indices,
        color="#D97732",
        opacity=1.0,
        name="Sherd Placement",
        hovertemplate=(
            "Estimated sherd placement"
            "<extra></extra>"
        ),
    )


# ============================================================
# CREATE HIGHLIGHTED REGION PATCH
# ============================================================

def create_region_patch(vessel_name, region_name):
    """Create a small green patch to highlight the estimated region."""

    dimensions = get_vessel_dimensions(vessel_name)
    regions = dimensions["regions"]

    if region_name not in regions:
        return None, None, None

    region_z, region_radius = regions[region_name]

    profile = get_vessel_profile(vessel_name)

    patch_width = min(
        region_radius * 0.65,
        1.4
    )

    patch_height = dimensions["height"] * 0.08

    x_values = np.linspace(
        -patch_width,
        patch_width,
        20
    )

    z_values = np.linspace(
        max(0, region_z - patch_height),
        min(
            dimensions["height"],
            region_z + patch_height
        ),
        12
    )

    X, Z = np.meshgrid(
        x_values,
        z_values
    )

    Y = np.zeros_like(X)

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):

            radius = radius_at_height(
                Z[row, col],
                profile
            )

            x_limited = np.clip(
                X[row, col],
                -radius * 0.95,
                radius * 0.95
            )

            Y[row, col] = -math.sqrt(
                max(
                    radius**2 - x_limited**2,
                    0
                )
            ) - 0.02

    return X, Y, Z


# ============================================================
# COMPLETE 3D VESSEL VISUALIZATION
# ============================================================

def create_3d_vessel(
    vessel_name,
    selected_region,
    contour
):
    """Create the interactive 3D vessel and sherd visualization."""

    (
        X,
        Y,
        Z,
        dimensions,
        profile,
    ) = create_vessel_surface(vessel_name)

    fig = go.Figure()

    # Vessel surface.
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[
                [0, "#A7A7A7"],
                [0.5, "#D0D0D0"],
                [1, "#EFEFEF"],
            ],
            opacity=0.55,
            showscale=False,
            name="Reference Vessel",
            hovertemplate=(
                "Reference vessel"
                "<extra></extra>"
            ),
        )
    )

    # Highlight selected region.
    X_patch, Y_patch, Z_patch = create_region_patch(
        vessel_name,
        selected_region
    )

    if X_patch is not None:
        fig.add_trace(
            go.Surface(
                x=X_patch,
                y=Y_patch,
                z=Z_patch,
                colorscale=[
                    [0, "#18A558"],
                    [1, "#18A558"],
                ],
                opacity=0.95,
                showscale=False,
                name=f"Estimated Region: {selected_region}",
                hovertemplate=(
                    f"Estimated region: {selected_region}"
                    "<extra></extra>"
                ),
            )
        )

    # Sherd placement.
    sherd_mesh = create_3d_sherd(
        contour,
        vessel_name,
        selected_region
    )

    if sherd_mesh is not None:
        fig.add_trace(sherd_mesh)

    # Region label.
    regions = dimensions["regions"]

    if selected_region in regions:
        region_z, region_radius = regions[selected_region]

        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[-region_radius - 0.35],
                z=[region_z],
                mode="text",
                text=[selected_region],
                textposition="middle center",
                name="Region Label",
                showlegend=False,
                textfont={
                    "size": 12,
                },
            )
        )

    fig.update_layout(
        title=f"3D Vessel Model — {vessel_name}",
        height=650,
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height",
            aspectmode="manual",
            aspectratio=dict(
                x=1,
                y=1,
                z=1.5
            ),
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=-2.0,
                    z=1.2
                )
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


# ============================================================
# COMPATIBILITY LEVEL HELPER
# ============================================================

def get_compatibility_level(score):
    """Convert a compatibility score into a descriptive level."""

    if score >= 80:
        return "High"
    elif score >= 65:
        return "Moderate"
    elif score >= 50:
        return "Low"

    return "Very Low"


# ============================================================
# NORMALIZE MATCHER OUTPUT
# ============================================================

def normalize_match_results(results):
    """
    Normalize the output of find_best_vessel.

    Expected output is a list of dictionaries containing
    vessel, region, score, compatibility_level, and description.
    """

    if results is None:
        return []

    if isinstance(results, dict):
        if "vessel" in results:
            results = [results]
        else:
            results = list(results.values())

    if not isinstance(results, (list, tuple)):
        return []

    normalized = []

    for item in results:
        if not isinstance(item, dict):
            continue

        try:
            score = float(item.get("score", 0))
        except (ValueError, TypeError):
            score = 0.0

        vessel = item.get(
            "vessel",
            item.get("vessel_name", "Unknown Vessel")
        )

        region = item.get(
            "region",
            item.get("region_name", "Unknown Region")
        )

        normalized.append({
            **item,
            "vessel": str(vessel),
            "region": str(region),
            "score": score,
            "compatibility_level": item.get(
                "compatibility_level",
                get_compatibility_level(score)
            ),
            "description": item.get(
                "description",
                "No description available."
            ),
            "reference_type": item.get(
                "reference_type",
                "Reference profile"
            ),
        })

    normalized.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return normalized


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏺 Pottery Analysis")

    st.write(
        "Upload a pottery sherd image to analyze its shape "
        "and compare its geometric features with predefined "
        "vessel profiles."
    )

    st.divider()

    st.subheader("Display Settings")

    show_intermediate_images = st.checkbox(
        "Show image processing results",
        value=True
    )

    show_mirror_analysis = st.checkbox(
        "Show mirror analysis",
        value=True
    )

    show_feature_details = st.checkbox(
        "Show detailed geometric features",
        value=True
    )

    show_all_matches = st.checkbox(
        "Show all vessel matches",
        value=True
    )

    st.divider()

    st.caption(
        "Prototype notice: vessel profiles and 3D dimensions "
        "are predefined reference values. They should not be "
        "treated as validated archaeological measurements."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏺 Pottery Sherd Analysis & Vessel Matching")

st.markdown(
    """
    Upload a pottery sherd image to examine its contour, extract
    geometric features, compare its shape with reference vessel
    profiles, and visualize the estimated matching region on a
    simplified 3D vessel model.
    """
)

st.info(
    "This application is a prototype. Compatibility scores "
    "indicate similarity to the configured reference profiles; "
    "they do not prove that a sherd originally belonged to a "
    "particular vessel."
)


# ============================================================
# PROFILE ERROR CHECK
# ============================================================

if PROFILE_ERRORS:
    st.error(
        "There are errors in the vessel reference profiles. "
        "Please check vessel_profiles.py."
    )

    with st.expander("View profile errors"):
        st.write(PROFILE_ERRORS)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a pottery sherd image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG."
)

if uploaded_file is None:
    st.info(
        "Upload an image to begin the analysis."
    )

    st.subheader("Analysis Workflow")

    workflow_columns = st.columns(5)

    workflow_steps = [
        ("1️⃣", "Image Upload"),
        ("2️⃣", "Background Removal"),
        ("3️⃣", "Contour Detection"),
        ("4️⃣", "Feature Matching"),
        ("5️⃣", "3D Visualization"),
    ]

    for column, (icon, label) in zip(
        workflow_columns,
        workflow_steps
    ):
        with column:
            st.markdown(f"### {icon}")
            st.write(label)

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:
    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as error:
    st.error(
        f"Unable to read the uploaded image: {error}"
    )
    st.stop()


# ============================================================
# IMAGE PROCESSING
# ============================================================

with st.spinner(
    "Removing background and detecting the sherd contour..."
):
    foreground_image, foreground_mask, contour = (
        remove_background(original_image)
    )

if (
    foreground_image is None
    or foreground_mask is None
    or contour is None
):
    st.error(
        "The application could not detect a suitable sherd "
        "contour. Try an image with a clear object boundary, "
        "good lighting, and a contrasting background."
    )

    st.subheader("Uploaded Image")
    st.image(
        original_image,
        caption="Original pottery sherd image",
        use_container_width=True
    )

    st.stop()


# ============================================================
# DRAW CONTOUR
# ============================================================

contour_image = draw_contour_on_image(
    original_image,
    contour
)


# ============================================================
# MIRROR ANALYSIS
# ============================================================

mirrored_mask = create_mirrored_mask(
    foreground_mask
)

symmetry_results = calculate_mirror_symmetry(
    foreground_mask,
    mirrored_mask
)

mirrored_contour = mirror_contour(
    contour
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

try:
    with st.spinner(
        "Extracting geometric features..."
    ):
        original_features = calculate_features(
            contour
        )

        mirrored_features = calculate_features(
            mirrored_contour
        )

except Exception as error:
    st.error(
        f"Feature extraction failed: {error}"
    )
    st.stop()


if not isinstance(original_features, dict):
    st.error(
        "The calculate_features() function must return "
        "a dictionary of geometric features."
    )
    st.stop()


if not original_features:
    st.error(
        "No geometric features were extracted from the image."
    )
    st.stop()


feature_comparison = create_feature_comparison(
    original_features,
    mirrored_features
)


# ============================================================
# VESSEL MATCHING
# ============================================================

try:
    with st.spinner(
        "Comparing the sherd with reference vessel profiles..."
    ):
        raw_match_results = find_best_vessel(
            original_features,
            VESSEL_PROFILES
        )

        match_results = normalize_match_results(
            raw_match_results
        )

except Exception as error:
    st.error(
        f"Vessel matching failed: {error}"
    )
    st.stop()


if not match_results:
    st.warning(
        "No vessel matches were returned. Check the reference "
        "profiles in vessel_profiles.py and the matching logic "
        "in matcher.py."
    )
    st.stop()


# Best result is the first item after sorting.
best_match = match_results[0]

best_vessel = best_match["vessel"]
best_region = best_match["region"]
best_score = best_match["score"]
best_level = best_match["compatibility_level"]
best_description = best_match["description"]
best_reference_type = best_match["reference_type"]


# ============================================================
# STREAMLIT TABS
# ============================================================

(
    overview_tab,
    image_tab,
    features_tab,
    matching_tab,
    visualization_tab,
) = st.tabs([
    "📊 Overview",
    "🖼️ Image Processing",
    "📐 Geometric Features",
    "🏺 Vessel Matching",
    "🧊 3D Visualization",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with overview_tab:

    st.subheader("Analysis Summary")

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric(
            "Best Matching Vessel",
            best_vessel
        )

    with metric_columns[1]:
        st.metric(
            "Compatibility Score",
            f"{best_score:.2f}%"
        )

    with metric_columns[2]:
        st.metric(
            "Estimated Region",
            best_region
        )

    with metric_columns[3]:
        st.metric(
            "Compatibility Level",
            best_level
        )

    st.divider()

    st.subheader("Best Match Description")

    st.write(best_description)

    st.write(
        f"**Reference type:** {best_reference_type}"
    )

    st.divider()

    st.subheader("Original Sherd")

    overview_columns = st.columns(2)

    with overview_columns[0]:
        st.image(
            original_image,
            caption="Uploaded pottery sherd",
            use_container_width=True
        )

    with overview_columns[1]:
        st.image(
            contour_image,
            caption="Detected outer contour",
            use_container_width=True
        )

    st.divider()

    st.subheader("Compatibility Ranking")

    ranking_df = pd.DataFrame([
        {
            "Rank": index + 1,
            "Vessel": item["vessel"],
            "Region": item["region"],
            "Compatibility Score (%)": round(
                item["score"],
                2
            ),
            "Level": item["compatibility_level"],
        }
        for index, item in enumerate(match_results)
    ])

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )

    # Native Streamlit chart; no file export.
    chart_df = ranking_df.set_index(
        "Vessel"
    )[["Compatibility Score (%)"]]

    st.bar_chart(
        chart_df,
        use_container_width=True
    )


# ============================================================
# TAB 2: IMAGE PROCESSING
# ============================================================

with image_tab:

    st.subheader("Image Processing Results")

    st.write(
        "The uploaded image is processed to separate the sherd "
        "from its background. The application then identifies "
        "the outer contour used for geometric analysis."
    )

    image_columns = st.columns(2)

    with image_columns[0]:
        st.image(
            original_image,
            caption="1. Original Image",
            use_container_width=True
        )

    with image_columns[1]:
        st.image(
            foreground_image,
            caption="2. Background Removed",
            use_container_width=True
        )

    st.divider()

    st.subheader("Detected Contour")

    st.image(
        contour_image,
        caption="Green line indicates the detected outer contour",
        use_container_width=True
    )

    st.divider()

    st.subheader("Foreground Mask")

    st.image(
        foreground_mask,
        caption="Binary mask used for shape analysis",
        clamp=True,
        use_container_width=True
    )

    if show_intermediate_images:
        with st.expander(
            "View additional image processing information",
            expanded=False
        ):
            image_height, image_width = foreground_mask.shape

            contour_area = cv2.contourArea(contour)
            contour_perimeter = cv2.arcLength(
                contour,
                True
            )

            x, y, width, height = cv2.boundingRect(
                contour
            )

            details_columns = st.columns(3)

            with details_columns[0]:
                st.metric(
                    "Image Width",
                    f"{image_width} px"
                )

            with details_columns[1]:
                st.metric(
                    "Image Height",
                    f"{image_height} px"
                )

            with details_columns[2]:
                st.metric(
                    "Contour Area",
                    f"{contour_area:.2f} px²"
                )

            st.write({
                "Bounding Box X": int(x),
                "Bounding Box Y": int(y),
                "Bounding Box Width": int(width),
                "Bounding Box Height": int(height),
                "Contour Perimeter": round(
                    contour_perimeter,
                    2
                ),
            })


# ============================================================
# TAB 3: GEOMETRIC FEATURES
# ============================================================

with features_tab:

    st.subheader("Extracted Geometric Features")

    st.write(
        "Geometric features describe the shape of the sherd. "
        "These values are compared with the reference features "
        "configured in the vessel profiles."
    )

    # Show selected important features when available.
    important_features = [
        "Area",
        "Perimeter",
        "Aspect Ratio",
        "Circularity",
        "Solidity",
        "Extent",
        "Convexity",
        "Elongation",
        "Eccentricity",
    ]

    available_features = {
        key: value
        for key, value in original_features.items()
        if key in important_features
    }

    if available_features:
        st.subheader("Key Features")

        feature_columns = st.columns(3)

        for index, (key, value) in enumerate(
            available_features.items()
        ):
            with feature_columns[index % 3]:
                if isinstance(
                    value,
                    (int, float, np.integer, np.floating)
                ):
                    st.metric(
                        key,
                        f"{float(value):.4f}"
                    )
                else:
                    st.metric(
                        key,
                        str(value)
                    )

    st.divider()

    if show_feature_details:
        st.subheader("Complete Feature Comparison")

        st.dataframe(
            feature_comparison,
            use_container_width=True,
            hide_index=True
        )

        with st.expander(
            "View original feature values",
            expanded=False
        ):
            original_feature_df = pd.DataFrame(
                list(original_features.items()),
                columns=[
                    "Feature",
                    "Value"
                ]
            )

            st.dataframe(
                original_feature_df,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.info(
            "Enable 'Show detailed geometric features' in "
            "the sidebar to view the complete feature table."
        )

    st.divider()

    st.subheader("Why Are Geometric Features Used?")

    st.markdown(
        """
        - **Aspect Ratio:** Describes the relationship between
          the width and height of the detected shape.
        - **Circularity:** Indicates how closely the contour
          resembles a circle.
        - **Solidity:** Compares the contour area with its
          convex hull area.
        - **Extent:** Measures how much of the bounding box
          is occupied by the contour.
        - **Convexity:** Describes the relationship between
          the contour perimeter and its convex hull perimeter.
        - **Elongation:** Describes how stretched the shape is.
        - **Eccentricity:** Describes the elongation of an
          ellipse fitted to the contour.
        """
    )


# ============================================================
# TAB 4: VESSEL MATCHING
# ============================================================

with matching_tab:

    st.subheader("Vessel Compatibility Analysis")

    st.write(
        "The extracted features are compared with predefined "
        "reference vessel profiles. The matching algorithm "
        "returns compatibility scores for the available "
        "vessel and region combinations."
    )

    result_columns = st.columns(3)

    with result_columns[0]:
        st.metric(
            "Best Vessel",
            best_vessel
        )

    with result_columns[1]:
        st.metric(
            "Best Score",
            f"{best_score:.2f}%"
        )

    with result_columns[2]:
        st.metric(
            "Estimated Region",
            best_region
        )

    st.divider()

    st.subheader("Best Match Details")

    st.write(
        f"**Vessel:** {best_vessel}"
    )

    st.write(
        f"**Region:** {best_region}"
    )

    st.write(
        f"**Compatibility:** {best_level}"
    )

    st.write(
        f"**Reference type:** {best_reference_type}"
    )

    st.write(best_description)

    st.divider()

    if show_all_matches:
        st.subheader("All Matching Results")

        all_results_df = pd.DataFrame([
            {
                "Rank": index + 1,
                "Vessel": item["vessel"],
                "Region": item["region"],
                "Score (%)": round(
                    item["score"],
                    2
                ),
                "Compatibility": item["compatibility_level"],
                "Description": item["description"],
            }
            for index, item in enumerate(match_results)
        ])

        st.dataframe(
            all_results_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Enable 'Show all vessel matches' in the sidebar "
            "to view the complete ranking."
        )

    st.divider()

    st.subheader("How to Interpret the Score")

    st.markdown(
        """
        The compatibility score represents the similarity
        calculated by the matching algorithm between the
        extracted sherd features and a configured reference
        profile.

        - **High:** Stronger geometric similarity.
        - **Moderate:** Some geometric similarity.
        - **Low:** Limited geometric similarity.
        - **Very Low:** Weak geometric similarity.

        A high score is not proof of an original physical fit.
        The result depends on the feature extraction process,
        matching method, and quality of the reference profiles.
        """
    )

# ============================================================
# TAB 5: 3D VISUALIZATION
# ============================================================

with visualization_tab:

    st.subheader("3D Vessel Reconstruction")

    st.write(
        "The 3D model shows a simplified parametric vessel profile. "
        "The green patch highlights the estimated matching region, "
        "and the orange mesh represents the uploaded sherd silhouette "
        "mapped approximately onto the vessel surface."
    )

    # Automatically use the best-matching vessel and region.
    selected_vessel = best_vessel
    selected_region = best_region

    st.markdown(f"**Best-Matching Vessel:** {selected_vessel}")
    st.markdown(f"**Estimated Matching Region:** {selected_region}")

    try:
        fig_3d = create_3d_vessel(
            selected_vessel,
            selected_region,
            contour
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True,
            key="pottery_3d_vessel"
        )

    except Exception as error:
        st.error(
            f"Unable to generate the 3D visualization: {error}"
        )

    st.divider()

    # Display model information without dropdowns.
    selected_dimensions = get_vessel_dimensions(selected_vessel)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Height",
            f"{selected_dimensions['height']:.2f} units"
        )

    with col2:
        st.metric(
            "Maximum Radius",
            f"{selected_dimensions['max_radius']:.2f} units"
        )

    with col3:
        st.metric(
            "Compatibility Score",
            f"{best_score:.2f}%"
        )

    st.caption(
        "The vessel is generated from predefined profile points. "
        "The highlighted region and sherd placement are estimates, "
        "not a verified physical fit."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pottery Sherd Analysis & Vessel Matching | "
    "Python • OpenCV • Streamlit • NumPy • Pandas • Plotly"
)

st.caption(
    "Prototype system using predefined vessel reference profiles. "
    "Results should be validated against real archaeological "
    "sherd images and vessel measurements before research use."
)
