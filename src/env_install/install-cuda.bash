#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
echo "--- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---"
echo "- Setting up the virtual environment"

# Set paths
VIRTUAL_ENV="$PROJECT_ROOT/python3"
PYTHON_EXE="python3.12"
export PIP_RETRIES=10
export PIP_TIMEOUT=180
PIP_CACHE_DIR="$VIRTUAL_ENV/.pip-cache"

# Check if Python is available
if ! command -v "$PYTHON_EXE" &> /dev/null
then
    echo "Python is not found in PATH. Please install Python 3.12 first."
    read -p "Press enter to exit..."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -f "$VIRTUAL_ENV/bin/activate" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_EXE" -m venv "$VIRTUAL_ENV"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        read -p "Press enter to exit..."
        exit 1
    fi
fi

# Activate virtual environment
echo "- Virtual environment activation"
source "$VIRTUAL_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    read -p "Press enter to exit..."
    exit 1
fi
PYTHON_EXE="$VIRTUAL_ENV/bin/python"

# Update pip
echo "- Updating the pip package"
"$PYTHON_EXE" -m pip install --upgrade pip -r "$SCRIPT_DIR/requirements-pip-options.txt" --cache-dir "$PIP_CACHE_DIR" --prefer-binary --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT"
if [ $? -ne 0 ]; then
    echo "Failed to update pip"
    read -p "Press enter to exit..."
    exit 1
fi

# TensorRT's small source front-end otherwise creates an isolated build
# environment and downloads wheel/setuptools again. Prepare those tools once.
install_build_requirements() {
    "$PYTHON_EXE" -m pip install "setuptools==78.1.0" "wheel>=0.45,<1" "packaging>=24,<27" -r "$SCRIPT_DIR/requirements-pip-options.txt" --cache-dir "$PIP_CACHE_DIR" --prefer-binary --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT"
}
echo "- Preparing Python package build tools"
if ! install_build_requirements; then
    echo "- Build tools download failed; retrying 1/2"
    sleep 3
    if ! install_build_requirements; then
        echo "- Build tools download failed; retrying 2/2"
        sleep 3
        if ! install_build_requirements; then
            echo "Failed to prepare Python package build tools after 3 attempts"
            read -p "Press enter to exit..."
            exit 1
        fi
    fi
fi

# Install requirements
echo
echo "- Installing the requirements"
if ! sudo apt-get install python3-tk wmctrl mesa-utils portaudio19-dev ffmpeg xdotool -y; then
    echo "Failed to install system requirements"
    read -p "Press enter to exit..."
    exit 1
fi
install_cuda_requirements() {
    "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements-cuda.txt" --cache-dir "$PIP_CACHE_DIR" --prefer-binary --no-build-isolation --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT"
}
if ! install_cuda_requirements; then
    echo "- CUDA requirements download failed; retrying 1/2"
    sleep 3
    if ! install_cuda_requirements; then
        echo "- CUDA requirements download failed; retrying 2/2"
        sleep 3
        if ! install_cuda_requirements; then
            echo "Failed to install CUDA requirements after 3 attempts"
            read -p "Press enter to exit..."
            exit 1
        fi
    fi
fi
if ! "$PYTHON_EXE" -c "import tensorrt as trt; raise SystemExit(0 if trt.__version__ == '10.14.1.48.post1' else 1)"; then
    echo "TensorRT installation validation failed"
    read -p "Press enter to exit..."
    exit 1
fi
if ! "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements.txt" --cache-dir "$PIP_CACHE_DIR" --prefer-binary --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT"; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
exit 0
