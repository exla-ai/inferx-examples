"""
RoboPoint model integration using Docker container.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import shutil

class RoboPoint:
    """
    RoboPoint model for keypoint detection in images.
    This implementation uses a Docker container to run the model.
    """
    
    def __init__(self, use_8bit=True, docker_image="exla/robopoint-gpu:latest"):
        """
        Initialize the RoboPoint model.
        
        Args:
            use_8bit (bool): Whether to use 8-bit quantization.
            docker_image (str): The Docker image to use.
        """
        self.use_8bit = use_8bit
        self.docker_image = docker_image
        
        # Check if Docker is installed
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("Docker is not installed or not in PATH. Please install Docker to use this model.")
        
        # Check if the Docker image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.docker_image], 
            capture_output=True, 
            text=True
        )
        
        if not result.stdout.strip():
            print(f"Docker image {self.docker_image} not found. Please build and push the image first.")
            print("You can build the image with: docker build -t exla/robopoint-gpu:latest -f docker/models/robopoint/gpu/Dockerfile .")
    
    def inference(self, image_path, text_instruction=None, output=None):
        """
        Run inference with the RoboPoint model to predict keypoint affordances.
        
        Args:
            image_path (str): Path to input image
            text_instruction (str, optional): Language instruction for the model
            output (str, optional): Path to save the output visualization
            
        Returns:
            dict: Results containing keypoints and their coordinates
        """
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if text_instruction is None:
            text_instruction = "Find keypoints in the image"
        
        # Create a temporary directory for output if not specified
        if output is None:
            temp_dir = tempfile.mkdtemp()
            output = os.path.join(temp_dir, "output.png")
        
        # Get absolute paths
        abs_image_path = os.path.abspath(image_path)
        abs_output_path = os.path.abspath(output)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        
        # Run Docker container
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{os.path.dirname(abs_image_path)}:/app/data/input",
            "-v", f"{os.path.dirname(abs_output_path)}:/app/data/output",
            self.docker_image,
            "inference",
            f"/app/data/input/{os.path.basename(abs_image_path)}",
            text_instruction,
            f"/app/data/output/{os.path.basename(abs_output_path)}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Inference completed. Output saved to {output}")
            return {"status": "success", "output_path": output}
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")
            return {"status": "error", "message": str(e)}

# Factory function to match the expected interface
def robopoint(**kwargs):
    """
    Factory function to create a RoboPoint model instance.
    
    Args:
        **kwargs: Additional arguments to pass to the model.
        
    Returns:
        A RoboPoint model instance.
    """
    return RoboPoint(**kwargs) 