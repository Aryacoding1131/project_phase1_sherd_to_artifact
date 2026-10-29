from transformers import pipeline
from PIL import Image


# ============================================================
# LOAD PRETRAINED AI MODEL
# ============================================================

def load_classifier():
    """
    Load a pretrained zero-shot image classification model.
    """

    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


# ============================================================
# CANDIDATE IMAGE DESCRIPTIONS
# ============================================================

CANDIDATE_LABELS = [
    "a photograph of a broken pottery sherd",
    "a photograph of a ceramic vessel fragment",
    "a photograph of a stone fragment",
    "a photograph of a brick fragment",
    "a photograph of a tile fragment",
    "a photograph of a shell fragment",
    "a photograph of an unidentified object"
]


# ============================================================
# POTTERY IDENTIFICATION
# ============================================================

_classifier = None


def validate_pottery_image(image):
    """
    Perform preliminary visual classification of an image.

    The returned scores are model scores, not calibrated
    probabilities that an object is pottery.
    """

    global _classifier

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    image = image.convert("RGB")

    # Load the model only when it is first required.
    if _classifier is None:
        _classifier = load_classifier()

    # Compare the image with the candidate descriptions.
    predictions = _classifier(
        image,
        candidate_labels=CANDIDATE_LABELS
    )

    # Obtain the highest-ranked description.
    best_prediction = predictions[0]

    predicted_label = best_prediction["label"]
    model_score = float(best_prediction["score"])

    # Group the pottery-related descriptions.
    pottery_labels = {
        "a photograph of a broken pottery sherd",
        "a photograph of a ceramic vessel fragment"
    }

    if predicted_label in pottery_labels:
        predicted_class = "Potential Pottery"
    else:
        predicted_class = "Other or Uncertain Object"

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "model_score": model_score,
        "all_predictions": predictions
    }
  
