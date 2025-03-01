# SAM2 Example

This example demonstrates how to use the Segment Anything 2 (SAM2) model for image and video segmentation on NVIDIA Jetson devices.

## Prerequisites

- NVIDIA Jetson device (Jetson AGX Orin, Jetson Orin NX, etc.)
- Python 3.8+
- EXLA SDK installed
- OpenCV (for video and camera processing)

## Setup

To use this example, you need the SAM2 model file:

1. Download the SAM2 model file from one of these sources:
   - [SAM2 Model Checkpoints on GitHub](https://github.com/facebookresearch/segment-anything-2#model-checkpoints)
   - [SAM2 Base Model on HuggingFace](https://huggingface.co/facebook/sam2-base/resolve/main/sam2_b.pth)

2. Place the model file in the cache directory:
   ```
   ~/.cache/exla/sam2/sam2_b.pth
   ```

## Usage

### Image Segmentation

To segment an image (truck.jpg):

```bash
python example_sam2.py --mode image
```

This will process the image and save the results to the `data/output_truck/` directory.

### Video Segmentation

To segment a video (f1.mp4):

```bash
python example_sam2.py --mode video
```

This will process the video and save the results to the `data/output_f1/` directory.

### Using Numpy Arrays as Input

The SAM2 model also supports using numpy arrays directly as input. This is demonstrated in the `example_sam2.py` script:

```bash
python example_sam2.py
```

This example shows:
1. How to load an image as a numpy array
2. How to pass the numpy array to the SAM2 model
3. How to process the segmentation masks returned by the model
4. How to create overlay visualizations by combining the original image with the masks

The example includes three different use cases:
- Example 1: Using point prompts with an image file
- Example 2: Using box prompts with a video file
- Example 3: Using point prompts with a numpy array

You can run individual examples by uncommenting the corresponding function calls at the end of the script.

### Live Camera Feed

To process a live camera feed:

```bash
python example_sam2.py --mode camera
```

Additional camera options:
```bash
# Specify camera ID (for multiple cameras)
python example_sam2.py --mode camera --camera-id 1

# Set processing duration (in seconds)
python example_sam2.py --mode camera --duration 60
```

This will process the camera feed and save the results to the `data/output_camera/` directory.

### Benchmarking

To run benchmarks for image processing:

```bash
python example_sam2.py --mode image --benchmark
```

Video and camera modes automatically include benchmarking information.

## Example Files

- `truck.jpg`: Sample image for segmentation
- `f1.mp4`: Sample video for segmentation

## Output

The script will generate the following outputs:

### For Images:
- `segmented_image.jpg`: The main output image with segmentation overlay
- `segmented_image_direct_overlay_*.png`: Direct overlays for each mask
- `segmented_image_output_mask_*.png`: Visualizations of each mask

### For Videos:
- `segmented_video.mp4`: The output video with segmentation overlay

### For Camera Feed:
- `camera_feed.mp4`: The recorded camera feed with segmentation overlay

## Performance Metrics

The script provides detailed performance metrics:

- Processing time per frame
- Average processing time
- Frames per second (FPS)
- Total processing time

These metrics are displayed in the console output and, for camera mode, overlaid on the video frames.

## Troubleshooting

If you encounter issues:

1. **Model file not found or corrupted**:
   - Make sure you've downloaded the correct model file
   - Verify the file is placed in `~/.cache/exla/sam2/sam2_b.pth`
   - Check that the file size is correct (several hundred MB)

2. **Video/camera processing fails**:
   - Video and camera processing typically require a local model
   - The script will automatically detect if it's using a server instead of a local model
   - Make sure the model file is properly downloaded

3. **Camera not found**:
   - Check that your camera is properly connected
   - Try a different camera ID with `--camera-id`
   - Ensure you have the necessary permissions to access the camera

4. **Import errors**:
   - Make sure the EXLA SDK is properly installed
   - Install OpenCV with `pip install opencv-python`

## License

This example is part of the EXLA SDK and is licensed under the same terms as the EXLA SDK. 