from transformers import pipeline
from PIL import Image

print("Loading AI vision model...")

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

print("Model loaded!")

image = Image.open("images/test.png")

print("\nAnalyzing image...")

results = classifier(image)

print("\nAI RESULT:")

for item in results[:5]:
    print(f"{item['label']} -> {item['score']:.2%}")