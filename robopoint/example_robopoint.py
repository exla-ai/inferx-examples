from inferx.models.robopoint import robopoint


model = robopoint()

model.inference(
    image_path="data/sink.jpg",
    text_instruction="Find a few spots within the vacant area on the rightmost white plate.",
    output="data/sink_output.png"
)


output = model.inference(
    image_path="data/stair.jpg",
    text_instruction="Identify several places in the unoccupied space on the stair in the middle.",
    output="data/stair_output.png"
)


