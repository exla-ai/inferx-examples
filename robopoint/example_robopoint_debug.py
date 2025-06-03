from inferx.models.robopoint import robopoint

# Initialize with verbose logging and longer timeout
model = robopoint(verbosity="info")

print("Model initialized successfully!")

# Test first inference
print("Running first inference...")
result1 = model.inference(
    image_path="data/sink.jpg",
    text_instruction="Find a few spots within the vacant area on the rightmost white plate.",
    output="data/sink_output.png"
)
print(f"First inference completed: {result1}")

# Test second inference
print("Running second inference...")
result2 = model.inference(
    image_path="data/stair.jpg",
    text_instruction="Identify several places in the unoccupied space on the stair in the middle.",
    output="data/stair_output.png"
)
print(f"Second inference completed: {result2}")

print("All done!") 