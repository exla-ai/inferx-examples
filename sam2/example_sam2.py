import os
import cv2
import numpy as np
import tempfile
import io
from inferx.models.sam2 import sam2

# Setup with image and video
data_dir = "data"
    
# Input files
image_path = os.path.join(data_dir, "truck.jpg")
video_path = os.path.join(data_dir, "f1.mp4")
    
# Output directories
image_output_dir = os.path.join(data_dir, "output_truck")
video_output_dir = os.path.join(data_dir, "output_f1")
numpy_output_dir = os.path.join(data_dir, "output_numpy")
    
# Ensure output directories exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(numpy_output_dir, exist_ok=True)
    
model = sam2()

# Example 1: Using point prompts and an image
def example_1():
    print("\n=== Example 1: Image Processing with Point Prompts ===")
    result = model.inference(
        input=str(image_path),
        output=str(image_output_dir),
        prompt={"points": [[900, 600]], "labels": [1]}  # Center point
    )

    print(f"Image processing result: {result['status']}")
    if "processing_time" in result:
        print(f"Processing took {result['processing_time']:.3f} seconds")

# Example 2: Using box prompts and video
def example_2():
    print("\n=== Example 2: Video Processing with Box Prompts ===")
    result = model.inference(
        input=str(video_path),
        output=str(video_output_dir),
        prompt={"box": [400, 400, 1400, 800]}  # [x1, y1, x2, y2]
    )

    print(f"Video processing result: {result['status']}")
    if "processing_time" in result:
        print(f"Processing took {result['processing_time']:.3f} seconds")

# Example 3: Using numpy array as input
def example_3():
    print("\n=== Example 3: Numpy Array Processing ===")
    
    # Load the image as a NumPy array using OpenCV
    input_image_path = os.path.join(data_dir, "truck.jpg")
    numpy_image = cv2.imread(input_image_path)
    if numpy_image is None:
        raise ValueError(f"Failed to load image from {input_image_path}")
    print(f"Loaded numpy image with shape: {numpy_image.shape}")

    # Directly pass the NumPy array to the improved inference method.
    # The inference method now converts it to a PIL image internally if needed.
    result = model.inference(
        input=numpy_image,
        output=str(numpy_output_dir),
        prompt={"points": [[900, 600]], "labels": [1]}  # Example point prompt
    )

    def overlay_mask_on_image(original_image, mask, alpha=0.5, color=(255, 0, 0)):
        """
        Overlay a binary mask on the original image.
        
        Args:
            original_image (numpy.ndarray): The original image.
            mask (numpy.ndarray): The binary mask.
            alpha (float): Transparency factor.
            color (tuple): BGR color for the mask overlay.
        
        Returns:
            numpy.ndarray: Image with mask overlay.
        """
        # Convert mask to uint8 if it's boolean
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        
        # Resize mask if dimensions don't match the original image
        if mask.shape[:2] != original_image.shape[:2]:
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create a colored mask image (default color is red in BGR)
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = color
        
        # Blend the original image with the colored mask
        overlay = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
        return overlay


    print(f"Numpy processing result: {result['status']}")
    if "processing_time" in result:
        print(f"Processing took {result['processing_time']:.3f} seconds")

    if result['status'] == 'success':
        # Check if masks are returned in the result
        if 'masks' in result and result['masks']:
            masks = result['masks']
            print(f"Received {len(masks)} mask(s) from the model")
            # Process each mask (typically there's just one)
            for i, mask in enumerate(masks):
                mask = np.array(mask)  # Ensure mask is a NumPy array
                print(f"Mask {i} shape: {mask.shape}")
                print(f"Original image shape: {numpy_image.shape}")

                # Create an overlay of the mask on the original image
                overlay_image = overlay_mask_on_image(numpy_image, mask, alpha=0.5, color=(255, 0, 0))  # Blue color in BGR
                overlay_path = os.path.join(numpy_output_dir, f"overlay_{i}.png")
                cv2.imwrite(overlay_path, overlay_image)
                print(f"Saved overlay image to {overlay_path}")

                # Save the original mask (scaled for visualization)
                mask_path = os.path.join(numpy_output_dir, f"mask_{i}_original.png")
                cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
                print(f"Saved original mask to {mask_path}")

                # Resize the mask to match the original image dimensions
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                          (numpy_image.shape[1], numpy_image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                resized_mask_path = os.path.join(numpy_output_dir, f"mask_{i}_resized.png")
                cv2.imwrite(resized_mask_path, resized_mask * 255)
                print(f"Saved resized mask to {resized_mask_path}")

                # Display diagnostic information about the mask
                print(f"Number of segmented pixels in original mask: {np.sum(mask > 0)}")
                print(f"Number of segmented pixels in resized mask: {np.sum(resized_mask > 0)}")
        else:
            # If no masks are found, check the output directory for mask files
            print("No masks found in the result. Checking output directory for mask files...")
            mask_files = [f for f in os.listdir(numpy_output_dir) if f.endswith('.png') and 'mask' in f]
            if mask_files:
                print(f"Found {len(mask_files)} mask file(s):")
                for mask_file in mask_files:
                    print(f" - {mask_file}")
            else:
                print("No mask files found in the output directory.")
                print(f"Result keys: {list(result.keys())}")
    elif 'error' in result:
        print(f"Error processing the image: {result.get('error', 'Unknown error')}")
    else:
        print("Unexpected result format. Check the model output.")


# example_1()
# example_2()
example_3()