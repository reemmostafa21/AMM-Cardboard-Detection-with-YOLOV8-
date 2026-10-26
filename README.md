# AMM Cardboard Detection — Local PyCharm Test Starter

This starter project is for **testing an already-trained custom YOLOv8 cardboard detector** on images, recorded videos, and a local webcam. It does not include the `.pt` model or the Roboflow dataset: copy your own verified files into the folders below.

## 1. Project structure

- `weights/yolov8n_best.pt` — put your custom trained checkpoint here (or pass a different path with `--weights`).
- `test_images/cardboard.jpg` — put a test photo here.
- `test_videos/cardboard.mp4` — put a short test video here.
- `dataset/data.yaml` — optional, only needed for formal labeled validation/test metrics.
- `src/check_environment.py` — prints versions and CUDA availability.
- `src/check_model.py` — verifies the weights expose exactly one class: `cardboard`.
- `src/predict_image.py` — one-image test.
- `src/predict_video.py` — recorded-video test.
- `src/predict_webcam.py` — live webcam test.
- `src/validate_dataset.py` — optional labeled test/validation run.

## 2. Set up Python and PyCharm on Windows

1. Install Python 3.11 or 3.12 (64-bit) and PyCharm.
2. In PyCharm choose **Open** and open this project folder.
3. Open **Settings → Project → Python Interpreter** and create a new local **Virtualenv** for the project. Ensure PyCharm's interpreter is the project's `.venv`, not a different global Python.
4. Open PyCharm's Terminal (not the Python Console) and check:

   ```powershell
   python --version
   python -m pip --version
   ```

5. Install dependencies into that exact interpreter:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   For basic inference, the CPU can run these scripts. If you have an NVIDIA GPU and want GPU inference, first select the correct Windows/CUDA command on the official PyTorch Get Started Locally page, run it in the PyCharm Terminal, then install `requirements.txt`. Verify with `check_environment.py`; do not assume CUDA is active just because an NVIDIA GPU is present.

## 3. Copy the model and check it BEFORE inference

1. From the Kaggle artifact ZIP, extract the desired `*_best.pt` file into `weights/`. This notebook upload itself does not contain those binary checkpoint files.
2. Prefer a checkpoint for which the notebook confirms the correct one-class cardboard dataset and which passes evaluation. Be careful with a model file made during an uncertain/bad run.
3. In PyCharm right-click `src/check_environment.py` → **Run**. Verify which Python executable is being used and whether CUDA is available.
4. Right-click `src/check_model.py` → **Run**. If your file is not named `yolov8n_best.pt`, edit its default path or run it with `--weights`.
5. Continue only if it prints `Model classes: ['cardboard']` and `PASS`. If you see classes like `person`, `car`, etc., the file is the original COCO checkpoint, not the correct custom cardboard model.

## 4. Test a still photo first

1. Put a JPG/PNG that is NOT part of the training images into `test_images/`.
2. Name it `cardboard.jpg` or edit the source path in the Run Configuration.
3. Run `src/predict_image.py`.
4. The annotated output is saved under `runs/local_test/image/`. The terminal also prints each detected cardboard box's confidence.
5. To show an on-screen window, add `--show` to **Run → Edit Configurations → Parameters**. You can also pass a specific file with `--source "C:\path\to\test.jpg"` and weights with `--weights "C:\path\to\best.pt"`.

## 5. Test a recorded video

1. Record a short video showing a cardboard box from several distances and angles. Include both easy and hard examples: partial occlusion, clutter, and different lighting.
2. Put it at `test_videos/cardboard.mp4` or pass `--source "C:\path\to\video.mp4"`.
3. Run `src/predict_video.py`. It uses streaming inference, saves an annotated video under `runs/local_test/video/`, and displays a window unless `--no-show` is passed.
4. Press `q` in the display window to stop. For a long video or limited RAM, keep streaming enabled (the script already does that).

## 6. Test the webcam

1. Close other applications that may be using the webcam (Teams, Zoom, browser camera tests, etc.).
2. Run `src/predict_webcam.py`. Camera index `0` is the default camera. To try another camera, pass `--camera 1` in the Run Configuration parameters.
3. Put a real cardboard box in the camera view, vary its position/distance, and observe if the bounding box stays on the box instead of the background.
4. Press `q` in the video window to exit. Add `--save` if you also want to save the annotated webcam session.

## 7. Optional formal metrics on a labeled test split

This needs the **whole labeled YOLO-format dataset** (images, labels, and `data.yaml`), not just the `.pt` file. Extract it under `dataset/`, confirm `data.yaml` describes only `cardboard`, and run `src/validate_dataset.py`. It defaults to `split=test`. If the YAML does not contain a valid test split, the script stops rather than silently using another split. If the paths in the Roboflow-generated YAML do not resolve after moving the folder, inspect the YAML paths and correct them relative to the dataset root before rerunning.

Keep the held-out test split untouched until the end; repeatedly changing the model based on test-set scores turns that test set into a tuning set.

## 8. Troubleshooting

- **`ModuleNotFoundError: ultralytics` / `cv2` / `torch`:** PyCharm is probably using another interpreter, or dependencies were installed elsewhere. Run `python -m pip show ultralytics opencv-python torch` in PyCharm's Terminal and compare its Python path with the script's `sys.executable`.
- **`FileNotFoundError` for `.pt`, image or video:** put the file in the specified folder, check the extension, or pass the full quoted Windows path. Avoid ambiguous paths and do not use Kaggle paths such as `/kaggle/working/...`.
- **Class safety check rejects model:** expected if you accidentally loaded `yolov8n.pt` or another checkpoint. Locate and verify the custom `*_best.pt` file; don't simply remove the check.
- **CUDA is `False`:** CPU inference should work. If you expected GPU acceleration, check the NVIDIA driver, use the correct PyTorch install command for your CUDA/Windows setup, restart PyCharm, and rerun `check_environment.py`.
- **GPU out-of-memory:** test one image, close GPU-heavy programs, use `yolov8n` if you have verified weights for it, and lower `imgsz` (e.g., 512). Do not change model weights during an inference test.
- **Camera won't open / black window:** try `--camera 1`, close apps holding the camera, check Windows camera privacy permissions, and try the built-in Windows Camera app. A USB camera index may differ from 0.
- **`cv2.imshow` / display error:** run without `--show` for images or pass `--no-show` for videos; check that `opencv-python` (not only `opencv-python-headless`) is installed in this environment.
- **No detections:** confirm model classes first, remove training images from the test folder, improve lighting/focus, get the whole box into frame, and review confidence. Lowering `--conf` (e.g. to `0.15`) can reveal weaker detections but may also increase false positives.
- **Good photo test but poor real-world performance:** this is not unusual. Vary object sizes, angles, backgrounds, illumination, motion blur, and partial occlusion; save failures and label them to build a representative held-out test set.
- **Training-related multiprocessing error on Windows:** inference scripts don't train. If you later add a training script, place the training call inside `if __name__ == "__main__":` and try `workers=0` as a diagnostic.

## 9. Important limitations

- This is a local inference/testing starter, not the complete autonomous manipulator system. It does not command the robot arm, perform grasp planning, calculate camera-to-robot coordinates, or provide collision safety.
- A successful webcam test demonstrates inference is running; it does not prove accuracy, real-time FPS on the Jetson, or safe robot operation.
- A `best.pt` result trained/evaluated on Kaggle's Tesla T4 does not predict the exact latency of a Windows computer or a Jetson. Measure the target hardware separately.
