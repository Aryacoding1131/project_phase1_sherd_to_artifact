from PIL import Image
from pottery_validation import validate_pottery_image

# Open the test image
image = Image.open(
    r"D:\Projects_2nd_year\pottery_match_new\Screenshot 2026-09-04 111351.png"
).convert("RGB")

# Run preliminary identification
result = validate_pottery_image(image)

# Display the results
print("\n========== POTTERY IDENTIFICATION ==========")

print("Predicted class:", result["predicted_class"])
print("Best-matching description:", result["predicted_label"])
print("Model score:", round(result["model_score"], 4))

# Display all candidate scores
print("\n========== ALL CANDIDATE SCORES ==========")

for prediction in result["all_predictions"]:
    print(
        prediction["label"],
        "->",
        round(prediction["score"], 4)
    )

# Find the maximum score
max_prediction = max(
    result["all_predictions"],
    key=lambda prediction: prediction["score"]
)

print("\n========== MAXIMUM SCORE ==========")
print("Highest-scoring label:", max_prediction["label"])
print("Maximum score:", round(max_prediction["score"], 4))
print("Maximum score (%):", round(max_prediction["score"] * 100, 2), "%")
