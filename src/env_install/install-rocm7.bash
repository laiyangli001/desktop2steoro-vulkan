#!/bin/bash
# sed -i 's/\r$//' *.bash #correct for linux
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
echo "--- Desktop2Stereo Installer (With ROCm7 for AMD GPUs.) ---"
echo "- Setting up the virtual environment"

# Set paths
VIRTUAL_ENV="$PROJECT_ROOT/python3"
PYTHON_EXE="python3.11"
export PIP_RETRIES=10
export PIP_TIMEOUT=180

# Check if Python is available
if ! command -v "$PYTHON_EXE" &> /dev/null
then
    echo "Python is not found in PATH. Please install Python 3.11 first."
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
PYTHON_EXE="$VIRTUAL_ENV/bin/python"

# Activate virtual environment
echo "- Virtual environment activation"
source "$VIRTUAL_ENV/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    read -p "Press enter to exit..."
    exit 1
fi

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
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
if ! "$PYTHON_EXE" -m pip install python_xlib -r "$SCRIPT_DIR/requirements-pip-options.txt" --no-cache-dir --retries 5 --timeout 120; then
    echo "Failed to install python_xlib"
    read -p "Press enter to exit..."
    exit 1
fi
if ! "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements-rocm7.txt" --no-cache-dir --retries 5 --timeout 120; then
    echo "Failed to install ROCm requirements"
    read -p "Press enter to exit..."
    exit 1
fi
if ! "$PYTHON_EXE" -m pip install -r "$SCRIPT_DIR/requirements.txt" --no-cache-dir --retries 5 --timeout 120; then
    echo "Failed to install requirements"
    read -p "Press enter to exit..."
    exit 1
fi

echo "Python environment deployed successfully."
read -p "Press enter to exit..."
exit 0
