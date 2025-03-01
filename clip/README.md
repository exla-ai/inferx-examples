# CLIP Example

This example demonstrates how to use the CLIP (Contrastive Language-Image Pretraining) model with the Exla SDK. CLIP is a powerful model that can understand both images and text, allowing you to find the best matching images for a given text description.

## Setup

The Exla SDK automatically detects your hardware and uses the appropriate CLIP implementation:
- For Jetson devices (Orin Nano, AGX Orin): GPU-accelerated implementation
- For other GPUs: Standard GPU implementation
- For CPU-only systems: CPU implementation

**No manual setup required!** The library automatically installs the necessary dependencies based on your hardware.

## Running the Example

```bash
# Install the exla-sdk package
pip install exla-sdk

# Run the example
python examples/clip/example_clip.py
```

This will run the CLIP model on the sample images in the `data` directory and match them against some example text queries.

## Using Your Own Images and Prompts

To use your own images and prompts, modify the example code:

```python
from exla.models.clip import clip
import json

model = clip()

# Get matches with your own images and prompts
matches = model.inference(
    image_paths=["path/to/your/image1.jpg", "path/to/your/image2.jpg"],
    text_queries=["your custom prompt 1", "your custom prompt 2", "your custom prompt 3"]
)

print(json.dumps(matches, indent=2))
```

## How It Works

The Exla SDK uses "extreme lazy loading" to provide a seamless experience:

1. **Hardware Detection**: The library automatically detects your hardware and selects the appropriate implementation.
2. **On-demand Dependencies**: Dependencies are only installed when needed, reducing installation footprint.
3. **Automatic Setup**: Environment variables and configurations are set up automatically.
4. **Graceful Fallbacks**: If GPU acceleration isn't available, the library falls back to CPU.

## Performance Optimization (Optional)

For Jetson devices, the library will work out of the box, but you can get optimal performance by:

1. **Using Python 3.10 (Strongly Recommended)**
   
   The NVIDIA PyTorch wheel for Jetson devices is specifically built for Python 3.10. Using other Python versions will result in CPU-only inference, which is significantly slower.

   ```bash
   # Create a Python 3.10 virtual environment
   python3.10 -m venv py310_env
   source py310_env/bin/activate
   
   # Install pip if needed
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python get-pip.py
   
   # Install exla-sdk
   pip install exla-sdk
   ```

2. **The library will automatically install the NVIDIA PyTorch wheel** when running on Python 3.10 on a Jetson device. If you want to install it manually:
   ```bash
   pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl numpy==1.26.4
   ```

3. **Performance Comparison:**
   - With Python 3.10 + NVIDIA PyTorch (GPU): ~6 seconds total, ~1.5 seconds for inference
   - With other Python versions (CPU): ~10 seconds total, ~4 seconds for inference

The library will automatically detect if these optimizations are available and provide guidance if they're not. 