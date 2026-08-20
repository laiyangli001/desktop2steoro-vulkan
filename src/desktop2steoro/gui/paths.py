import os


GUI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(GUI_DIR)
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "desktop2stereo.log")
GUI_READY_FILE = os.path.join(LOG_DIR, "gui_ready.flag")
STOP_REQUEST_FILE = os.path.join(LOG_DIR, "stop.request")

# Kept as an alias for code that still references the old diagnostic log name.
DIAG_LOG = LOG_FILE
