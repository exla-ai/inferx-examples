# CLIP Example

This example demonstrates how to use the CLIP (Contrastive Language-Image Pretraining) model with the Exla SDK. CLIP is a powerful model that can understand both images and text, allowing you to find the best matching images for a given text description.


## Running the Example

```bash
pip install inferx

# Run the example
python clip/example_clip.py
```

This will run the CLIP model on the sample images in the `data` directory and match them against some example text queries.

## Using Your Own Images and Prompts

To use your own images and prompts, modify the example code:

```python
from inferx.models.clip import clip
import json

model = clip()

# Get matches with your own images and prompts
matches = model.inference(
    image_paths=["path/to/your/image1.jpg", "path/to/your/image2.jpg"],
    text_queries=["your custom prompt 1", "your custom prompt 2", "your custom prompt 3"]
)

print(json.dumps(matches, indent=2))
```


