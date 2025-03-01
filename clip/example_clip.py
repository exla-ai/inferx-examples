from exla.models.clip import clip
import json

model = clip()

# Get matches
matches = model.inference(
    image_paths=["data/dog.png", "data/cat.png"],
    text_queries=["a photo of a dog", "a photo of a cat", "a photo of a bird"]
)

print(json.dumps(matches, indent=2))
