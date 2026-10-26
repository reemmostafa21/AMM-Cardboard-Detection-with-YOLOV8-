"""Check which Python environment and ML packages PyCharm is running."""
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except Exception as exc:
    print("OpenCV import FAILED:", repr(exc))

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU inference will still work; training may be slow.")
except Exception as exc:
    print("PyTorch import FAILED:", repr(exc))

try:
    import ultralytics
    print("Ultralytics version:", ultralytics.__version__)
except Exception as exc:
    print("Ultralytics import FAILED:", repr(exc))
