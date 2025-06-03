# InferX Examples

This repository contains example code for using the InferX, a powerful toolkit for optimizing and deploying machine learning models across various hardware platforms.

## 🚀 Quick Start

1. **Add your user to the Docker group**
   ```bash
   sudo usermod -aG docker $USER
   ```
   **⚠️ Restart your terminal** for this to take effect.

2. **Install UV**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Clone the examples repository**:
   ```bash
   git clone https://github.com/exla-ai/inferx-examples.git
   cd inferx-examples
   ```

4. **Install InferX**:
   ```bash
   uv pip install git+https://github.com/exla-ai/InferX.git
   ```

5. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

6. **Run your first example**:
   ```bash
   python clip/example_clip.py
   ```

🎉 **That's it!** InferX will automatically detect your hardware and run an optimized instance of the model. 

