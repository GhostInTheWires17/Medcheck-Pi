# Medcheck Pi — Civic-grade Counterfeit Medicine Detector

Short: A PyQt6 GUI optimized for Raspberry Pi touchscreen that captures images, runs AI detection, stores results, and can export reports.

Quick links
- Source: [c:\repo\Medcheck Pi.py](c:\repo\Medcheck Pi.py)
- Important classes:
  - GUI / app: [`MainWindow`](c:\repo\Medcheck Pi.py)
  - DB: [`DatabaseManager`](c:\repo\Medcheck Pi.py)
  - Camera thread: [`CameraThread`](c:\repo\Medcheck Pi.py)
  - AI thread: [`AIDetectionThread`](c:\repo\Medcheck Pi.py)
  - Resource helper: [`ResourceManager`](c:\repo\Medcheck Pi.py)

Requirements
- Install Python 3.10+ (recommended).
- Install dependencies:
  - pip install -r requirements.txt

Notes:
- Many packages are optional depending on target environment:
  - On a Raspberry Pi with camera and GPIO use `picamera2` and `RPi.GPIO`.
  - For real AI inference install `torch` and `ultralytics` and provide a YOLO model (code uses "yolov8n.pt" by default).
  - If running without hardware, the app falls back to simulated resources and sample images.
- Resource files (fonts, sample images) are created/checked by [`ResourceManager`](c:\repo\Medcheck Pi.py) at startup.

Run
- From the repo root:
  - python "Medcheck Pi.py"

Database
- SQLite DB file path is set by the `DB_PATH` constant in the main script (default: Counterfeit_medicine_detector.db).
- See [`DatabaseManager`](c:\repo\Medcheck Pi.py) for schema and methods.

Exporting & Sync
- PDF export is handled by `reportlab`. If missing, PDF export is disabled.
- Network sync uses `requests` and can be toggled in Settings.

Hardware
- Buzzer uses GPIO pin defined as `BUZZER_PIN` in the script.
- Touch calibration handled by [`TouchCalibrationWidget`](c:\repo\Medcheck Pi.py).

Troubleshooting
- If PyQt6 installation fails, follow PyQt6 platform-specific docs.
- On RPi you may need system packages (alsa, portaudio, libatlas, etc.) before pip installing torch/pyaudio.

License / Disclaimer
- This project is for educational/demo purposes. See the footer string in PDF exporter for usage guidance.

If you want, I can:
- Pin specific package versions after you test on your device.
- Generate a minimal extras/optional requirements file (e.g., requirements-hw.txt) for RPi.
```# filepath: README.md
# Medcheck Pi — Civic-grade Counterfeit Medicine Detector

Short: A PyQt6 GUI optimized for Raspberry Pi touchscreen that captures images, runs AI detection, stores results, and can export reports.

Quick links
- Source: [c:\repo\Medcheck Pi.py](c:\repo\Medcheck Pi.py)
- Important classes:
  - GUI / app: [`MainWindow`](c:\repo\Medcheck Pi.py)
  - DB: [`DatabaseManager`](c:\repo\Medcheck Pi.py)
  - Camera thread: [`CameraThread`](c:\repo\Medcheck Pi.py)
  - AI thread: [`AIDetectionThread`](c:\repo\Medcheck Pi.py)
  - Resource helper: [`ResourceManager`](c:\repo\Medcheck Pi.py)

Requirements
- Install Python 3.10+ (recommended).
- Install dependencies:
  - pip install -r requirements.txt

Notes:
- Many packages are optional depending on target environment:
  - On a Raspberry Pi with camera and GPIO use `picamera2` and `RPi.GPIO`.
  - For real AI inference install `torch` and `ultralytics` and provide a YOLO model (code uses "yolov8n.pt" by default).
  - If running without hardware, the app falls back to simulated resources and sample images.
- Resource files (fonts, sample images) are created/checked by [`ResourceManager`](c:\repo\Medcheck Pi.py) at startup.

Run
- From the repo root:
  - python "Medcheck Pi.py"

Database
- SQLite DB file path is set by the `DB_PATH` constant in the main script (default: Counterfeit_medicine_detector.db).
- See [`DatabaseManager`](c:\repo\Medcheck Pi.py) for schema and methods.

Exporting & Sync
- PDF export is handled by `reportlab`. If missing, PDF export is disabled.
- Network sync uses `requests` and can be toggled in Settings.

Hardware
- Buzzer uses GPIO pin defined as `BUZZER_PIN` in the script.
- Touch calibration handled by [`TouchCalibrationWidget`](c:\repo\Medcheck Pi.py).

Troubleshooting
- If PyQt6 installation fails, follow PyQt6 platform-specific docs.
- On RPi you may need system packages (alsa, portaudio, libatlas, etc.) before pip installing torch/pyaudio.

License / Disclaimer
- This project is for educational/demo purposes. See the footer string in PDF exporter for usage guidance.

If you want, I can:
- Pin specific package versions after you test on your device.
- Generate a minimal extras/optional requirements file (e.g., requirements-hw.txt) for RPi.
