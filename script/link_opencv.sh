#!/bin/bash

# Ensure script is run inside Pixi environment
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "❌ Error: This script must be run via 'pixi run' or inside 'pixi shell'."
    exit 1
fi

echo "======================================================="
echo "🔗 OPENCV SYSTEM LINKER (Hybrid Mode)"
echo "======================================================="

# 1. Determine target location in Pixi Environment
SITE_PACKAGES="$CONDA_PREFIX/lib/python3.10/site-packages"
PIXI_CV2_DIR="$SITE_PACKAGES/cv2"
PIXI_CV2_SO="$SITE_PACKAGES/cv2.so"

# 2. Check if there's an OpenCV version installed via PIP/Conda
if [[ -d "$PIXI_CV2_DIR" ]] || [[ -f "$PIXI_CV2_SO" ]]; then
    echo "⚠️  Detected OpenCV installed via Pixi/Pip."
    echo "   Please remove 'opencv-contrib-python' from pixi.toml and run 'pixi install' first."
    echo "   Or run: pip uninstall -y opencv-python opencv-contrib-python"
    
    read -p "   Do you want me to uninstall the Pip version now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip uninstall -y opencv-python opencv-contrib-python
    else
        echo "❌ Cancelled. Cannot create link if OpenCV Pip is still present."
        exit 1
    fi
fi

# 3. Search for compiled OpenCV .so file in System (/usr/local)
# Filename pattern usually: cv2.cpython-310-x86_64-linux-gnu.so
echo "🔍 Searching for System OpenCV (CUDA build)..."

# Search in standard manual installation path
SYSTEM_CV2=$(find /usr/local/lib -name "cv2.cpython-310*.so" 2>/dev/null | head -n 1)

if [[ -z "$SYSTEM_CV2" ]]; then
    echo "❌ FAILED: Cannot find file cv2.cpython-310*.so in /usr/local/lib."
    echo "   Make sure you have run 'sudo make install' and System Python version = Pixi Python (3.10)."
    exit 1
fi

echo "✅ Found: $SYSTEM_CV2"

# 4. Create Symlink
echo "🔗 Creating Symlink to: $PIXI_CV2_SO"
ln -sf "$SYSTEM_CV2" "$PIXI_CV2_SO"

# 5. Quick Verification
echo "🧪 Verifying Import..."
python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}'); print(f'CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Pixi is now using System OpenCV (CUDA)."
else
    echo "❌ Failed to import cv2. Check Python version compatibility."
fi