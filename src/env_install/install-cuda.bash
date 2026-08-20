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
"$PYTHON_EXE" -m pip install --upgrade pip -r "$SCRIPT_DIR/requirements-pip-options.txt" --no-cache-dir --retries 5 --timeout 120
if [ $? -ne 0 ]; then
    echo "Failed to update pip"
    read -p "Press enter to exit..."
    exit 1
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
    "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements-cuda.txt" --no-cache-dir --retries 5 --timeout 120
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
if ! "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir --retries 5 --timeout 120; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
exit 0
