REFERENCE_FEATURES = [
    "Aspect Ratio",
    "Circularity",
    "Solidity",
    "Extent",
    "Convexity",
    "Elongation",
    "Eccentricity"
]


# ============================================================
# VESSEL PROFILES
# ============================================================

VESSEL_PROFILES = {

    # ========================================================
    # A - SMALL BOWL
    # ========================================================

    "A - Small Bowl": {

        "description":
            "Small open bowl with a relatively wide mouth.",

        "reference_type":
            "Provisional geometric reference",

        "regions": {

            "Rim": {
                "Aspect Ratio": 1.25,
                "Circularity": 0.82,
                "Solidity": 0.94,
                "Extent": 0.72,
                "Convexity": 0.92,
                "Elongation": 1.35,
                "Eccentricity": 0.67
            },

            "Upper Body": {
                "Aspect Ratio": 1.05,
                "Circularity": 0.84,
                "Solidity": 0.96,
                "Extent": 0.76,
                "Convexity": 0.94,
                "Elongation": 1.20,
                "Eccentricity": 0.55
            },

            "Lower Body": {
                "Aspect Ratio": 0.90,
                "Circularity": 0.86,
                "Solidity": 0.97,
                "Extent": 0.79,
                "Convexity": 0.95,
                "Elongation": 1.10,
                "Eccentricity": 0.42
            },

            "Base": {
                "Aspect Ratio": 1.15,
                "Circularity": 0.86,
                "Solidity": 0.98,
                "Extent": 0.81,
                "Convexity": 0.96,
                "Elongation": 1.12,
                "Eccentricity": 0.45
            }
        }
    },


    # ========================================================
    # B - STORAGE POT
    # ========================================================

    "B - Storage Pot": {

        "description":
            "Large storage vessel with a broad central body.",

        "reference_type":
            "Provisional geometric reference",

        "regions": {

            "Rim": {
                "Aspect Ratio": 1.30,
                "Circularity": 0.80,
                "Solidity": 0.93,
                "Extent": 0.70,
                "Convexity": 0.91,
                "Elongation": 1.42,
                "Eccentricity": 0.71
            },

            "Neck": {
                "Aspect Ratio": 0.75,
                "Circularity": 0.78,
                "Solidity": 0.94,
                "Extent": 0.73,
                "Convexity": 0.92,
                "Elongation": 1.48,
                "Eccentricity": 0.74
            },

            "Shoulder": {
                "Aspect Ratio": 1.05,
                "Circularity": 0.81,
                "Solidity": 0.91,
                "Extent": 0.70,
                "Convexity": 0.88,
                "Elongation": 1.30,
                "Eccentricity": 0.64
            },

            "Body": {
                "Aspect Ratio": 0.83,
                "Circularity": 0.84,
                "Solidity": 0.96,
                "Extent": 0.78,
                "Convexity": 0.94,
                "Elongation": 1.18,
                "Eccentricity": 0.53
            },

            "Base": {
                "Aspect Ratio": 1.15,
                "Circularity": 0.85,
                "Solidity": 0.98,
                "Extent": 0.82,
                "Convexity": 0.97,
                "Elongation": 1.14,
                "Eccentricity": 0.48
            }
        }
    },


    # ========================================================
    # C - CARINATED VASE
    # ========================================================

    "C - Carinated Vase": {

        "description":
            "Vase with a narrow neck and pronounced angular shoulder.",

        "reference_type":
            "Provisional geometric reference",

        "regions": {

            "Rim": {
                "Aspect Ratio": 1.20,
                "Circularity": 0.81,
                "Solidity": 0.94,
                "Extent": 0.71,
                "Convexity": 0.92,
                "Elongation": 1.32,
                "Eccentricity": 0.65
            },

            "Neck": {
                "Aspect Ratio": 0.62,
                "Circularity": 0.76,
                "Solidity": 0.92,
                "Extent": 0.68,
                "Convexity": 0.89,
                "Elongation": 1.72,
                "Eccentricity": 0.81
            },

            "Shoulder": {
                "Aspect Ratio": 1.10,
                "Circularity": 0.76,
                "Solidity": 0.87,
                "Extent": 0.66,
                "Convexity": 0.83,
                "Elongation": 1.35,
                "Eccentricity": 0.67
            },

            "Body": {
                "Aspect Ratio": 0.82,
                "Circularity": 0.83,
                "Solidity": 0.95,
                "Extent": 0.76,
                "Convexity": 0.92,
                "Elongation": 1.22,
                "Eccentricity": 0.57
            },

            "Base": {
                "Aspect Ratio": 1.12,
                "Circularity": 0.86,
                "Solidity": 0.98,
                "Extent": 0.82,
                "Convexity": 0.97,
                "Elongation": 1.12,
                "Eccentricity": 0.46
            }
        }
    },


    # ========================================================
    # D - CORRUGATED BOWL
    # ========================================================

    "D - Corrugated Bowl": {

        "description":
            "Wide bowl with a curved or irregular wall profile.",

        "reference_type":
            "Provisional geometric reference",

        "regions": {

            "Rim": {
                "Aspect Ratio": 1.32,
                "Circularity": 0.75,
                "Solidity": 0.88,
                "Extent": 0.67,
                "Convexity": 0.84,
                "Elongation": 1.40,
                "Eccentricity": 0.69
            },

            "Upper Body": {
                "Aspect Ratio": 1.15,
                "Circularity": 0.78,
                "Solidity": 0.90,
                "Extent": 0.70,
                "Convexity": 0.86,
                "Elongation": 1.28,
                "Eccentricity": 0.62
            },

            "Lower Body": {
                "Aspect Ratio": 0.95,
                "Circularity": 0.81,
                "Solidity": 0.93,
                "Extent": 0.75,
                "Convexity": 0.90,
                "Elongation": 1.16,
                "Eccentricity": 0.51
            },

            "Base": {
                "Aspect Ratio": 1.18,
                "Circularity": 0.84,
                "Solidity": 0.97,
                "Extent": 0.80,
                "Convexity": 0.95,
                "Elongation": 1.14,
                "Eccentricity": 0.48
            }
        }
    },


    # ========================================================
    # E - TALL VESSEL
    # ========================================================

    "E - Tall Vessel": {

        "description":
            "Tall vessel characterized by an elongated body.",

        "reference_type":
            "Provisional geometric reference",

        "regions": {

            "Rim": {
                "Aspect Ratio": 1.22,
                "Circularity": 0.82,
                "Solidity": 0.95,
                "Extent": 0.73,
                "Convexity": 0.93,
                "Elongation": 1.34,
                "Eccentricity": 0.66
            },

            "Neck": {
                "Aspect Ratio": 0.58,
                "Circularity": 0.77,
                "Solidity": 0.94,
                "Extent": 0.71,
                "Convexity": 0.92,
                "Elongation": 1.82,
                "Eccentricity": 0.84
            },

            "Body": {
                "Aspect Ratio": 0.68,
                "Circularity": 0.82,
                "Solidity": 0.95,
                "Extent": 0.76,
                "Convexity": 0.93,
                "Elongation": 1.60,
                "Eccentricity": 0.78
            },

            "Base": {
                "Aspect Ratio": 1.12,
                "Circularity": 0.86,
                "Solidity": 0.98,
                "Extent": 0.82,
                "Convexity": 0.97,
                "Elongation": 1.12,
                "Eccentricity": 0.46
            }
        }
    }
}


# ============================================================
# VALIDATE PROFILES
# ============================================================

def validate_vessel_profiles():
    """
    Check that every vessel region contains all required
    geometric features and that their values are valid.
    """

    errors = []

    for vessel_name, vessel in VESSEL_PROFILES.items():

        if "regions" not in vessel:
            errors.append(
                f"{vessel_name}: missing regions"
            )
            continue

        for region_name, region in vessel["regions"].items():

            for feature in REFERENCE_FEATURES:

                if feature not in region:
                    errors.append(
                        f"{vessel_name} / {region_name}: "
                        f"missing {feature}"
                    )
                    continue

                value = region[feature]

                if not isinstance(value, (int, float)):
                    errors.append(
                        f"{vessel_name} / {region_name}: "
                        f"{feature} must be numeric"
                    )
                    continue

                if value != value or value in (float("inf"), float("-inf")):
                    errors.append(
                        f"{vessel_name} / {region_name}: "
                        f"{feature} must be finite"
                    )

    return errors


PROFILE_ERRORS = validate_vessel_profiles()
