"""
Example usage of the Docker-based RoboPoint model.
"""

from examples.models.robopoint.robopoint import robopoint

# Initialize the model with the Docker image
model = robopoint(docker_image="exla/robopoint-gpu:latest")

# Run inference on an image
result = model.inference(
    image_path="examples/robopoint/data/sink.jpg",
    text_instruction="Find a few spots within the vacant area on the rightmost white plate.",
    output="examples/robopoint/data/sink_output.png"
)

print(f"Inference result: {result}")

# Run another inference
result = model.inference(
    image_path="examples/robopoint/data/stair.jpg",
    text_instruction="Identify several places in the unoccupied space on the stair in the middle.",
    output="examples/robopoint/data/stair_output.png"
)

print(f"Inference result: {result}")

# Example of running without specifying an output path
# The output will be saved to a temporary directory
result = model.inference(
    image_path="examples/robopoint/data/sink.jpg",
    text_instruction="Find keypoints in the image"
)

print(f"Inference result with auto-generated output path: {result}") 