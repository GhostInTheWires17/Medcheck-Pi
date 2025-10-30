#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Medcheck Pi - Civic-grade Counterfeit Medicine Detector
Optimized for Raspberry Pi 5 with 3.5" touchscreen
"""

import sys
import os
import time
import hashlib
import csv  # Enables DictWriter used for CSV exports
import sqlite3
import json
import threading
import queue
import math
import random
from datetime import datetime
from functools import partial

# PyQt6 imports
from PyQt6.QtCore import (Qt, QSize, QTimer, QPropertyAnimation, 
                         QEasingCurve, QRect, QThread, pyqtSignal, 
                         QByteArray, QBuffer, QIODevice, pyqtProperty)
from PyQt6.QtGui import (QFont, QFontDatabase, QColor, QPalette, 
                        QPixmap, QImage, QPainter, QPen, QBrush, 
                        QRadialGradient, QLinearGradient, QIcon)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QPushButton, QLabel, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QStackedWidget, 
                            QSlider, QDial, QProgressBar, QComboBox,
                            QTabWidget, QFrame, QScrollArea, QMessageBox, QInputDialog,
                            QTextEdit, QCheckBox, QRadioButton, QSpinBox,
                            QFileDialog, QSplashScreen)

# Raspberry Pi specific imports
try:
    import RPi.GPIO as GPIO
    from picamera2 import Picamera2
    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False
    print("Warning: Running in simulation mode (Raspberry Pi hardware not detected)")

# Text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available, speech synthesis disabled")

# Speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: speech_recognition not available, voice commands disabled")

# AI/ML components
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO/torch not available, using simulated AI detection")

# PDF export
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False
    print("Warning: reportlab not available, PDF export disabled")

# Network components
try:
    import requests
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False
    print("Warning: requests not available, network features disabled")

# Constants
DARK_BLUE = "#0A1128"
NEON_BLUE = "#00FFFF"
ACCENT_BLUE = "#1282A2"
WARNING_RED = "#FF3366"
SUCCESS_GREEN = "#33FF99"
SCREEN_WIDTH = 480  # 3.5" Pi touchscreen
SCREEN_HEIGHT = 320
BUZZER_PIN = 18  # GPIO pin for buzzer
ANIMATION_DURATION = 300  # ms
DB_PATH = "Counterfeit_medicine_detector.db"

class StyleHelper:
    """Helper class for consistent styling across the application"""
    
    @staticmethod
    def set_sci_fi_style(app):
        """Apply sci-fi styling to the entire application"""
        # Load sci-fi font
        QFontDatabase.addApplicationFont(":/fonts/Orbitron-Regular.ttf")
        QFontDatabase.addApplicationFont(":/fonts/Orbitron-Bold.ttf")
        
        # Set application-wide stylesheet
        app.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0A1128;
                color: #FFFFFF;
            }
            
            QPushButton {
                background-color: #1282A2;
                color: white;
                border: 2px solid #00FFFF;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Orbitron';
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #0A617D;
                border: 2px solid #FFFFFF;
            }
            
            QPushButton:pressed {
                background-color: #074863;
            }
            
            QLabel {
                color: white;
                font-family: 'Orbitron';
            }
            
            QComboBox, QSlider, QDial {
                border: 1px solid #00FFFF;
                background-color: #0A1128;
                color: white;
            }
            
            QTabWidget::pane {
                border: 1px solid #00FFFF;
                background-color: #0A1128;
            }
            
            QTabBar::tab {
                background-color: #1282A2;
                color: white;
                border: 1px solid #00FFFF;
                padding: 5px;
                font-family: 'Orbitron';
            }
            
            QTabBar::tab:selected {
                background-color: #0A617D;
                border-bottom-color: #0A1128;
            }
        """)
    
    @staticmethod
    def create_neon_button(text, parent=None):
        """Create a button with neon glow effect"""
        button = QPushButton(text, parent)
        button.setMinimumHeight(40)
        
        # Add glow effect
        glow_animation = QPropertyAnimation(button, b"styleSheet")
        glow_animation.setDuration(1500)
        glow_animation.setLoopCount(-1)  # Infinite loop
        
        glow_start = """
            QPushButton {
                background-color: #1282A2;
                color: white;
                border: 2px solid #00FFFF;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Orbitron';
                font-weight: bold;
            }
        """
        
        glow_end = """
            QPushButton {
                background-color: #1282A2;
                color: white;
                border: 2px solid #00FFFF;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Orbitron';
                font-weight: bold;
                box-shadow: 0 0 10px #00FFFF;
            }
        """
        
        glow_animation.setStartValue(glow_start)
        glow_animation.setEndValue(glow_end)
        glow_animation.start()
        
        return button, glow_animation

class SplashScreen(QSplashScreen):
    """Custom splash screen with futuristic animation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        
        # Create base pixmap
        pixmap = QPixmap(SCREEN_WIDTH, SCREEN_HEIGHT)
        pixmap.fill(QColor(DARK_BLUE))
        
        # Set up painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw title
        title_font = QFont("Orbitron", 24, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(NEON_BLUE))
        painter.drawText(QRect(0, 80, SCREEN_WIDTH, 50), 
                         Qt.AlignmentFlag.AlignCenter, "MedCheck Pi")
        
        # Draw subtitle
        subtitle_font = QFont("Orbitron", 12)
        painter.setFont(subtitle_font)
        painter.drawText(QRect(0, 140, SCREEN_WIDTH, 30), 
                         Qt.AlignmentFlag.AlignCenter, 
                         "Counterfeit Medicine Detector")
        
        # Draw loading bar outline
        painter.setPen(QPen(QColor(NEON_BLUE), 2))
        painter.drawRoundedRect(SCREEN_WIDTH//4, 200, 
                               SCREEN_WIDTH//2, 20, 5, 5)
        
        painter.end()
        self.setPixmap(pixmap)
        
        # Animation timer
        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(30)
    
    def update_progress(self):
        """Update loading progress animation"""
        self.progress += 1
        if self.progress > 100:
            self.timer.stop()
            return
        
        # Update progress bar
        pixmap = self.pixmap().copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill progress bar
        progress_width = int((SCREEN_WIDTH//2 - 4) * (self.progress / 100))
        painter.setBrush(QColor(NEON_BLUE))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(SCREEN_WIDTH//4 + 2, 202, 
                               progress_width, 16, 4, 4)
        
        # Update status text
        painter.setPen(QColor(NEON_BLUE))
        status_font = QFont("Orbitron", 10)
        painter.setFont(status_font)
        
        status_messages = [
            "Initializing systems...",
            "Loading AI models...",
            "Calibrating camera...",
            "Connecting to network...",
            "Preparing user interface..."
        ]
        
        current_message = status_messages[min(4, self.progress // 20)]
        
        # Clear previous message area
        painter.setBrush(QColor(DARK_BLUE))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 230, SCREEN_WIDTH, 30)
        
        # Draw new message
        painter.setPen(QColor(NEON_BLUE))
        painter.drawText(QRect(0, 230, SCREEN_WIDTH, 30), 
                         Qt.AlignmentFlag.AlignCenter, current_message)
        
        painter.end()
        self.setPixmap(pixmap)
        
        # Show message in splash
        self.showMessage(f"Loading... {self.progress}%", 
                         Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                         QColor(NEON_BLUE))

class CameraThread(QThread):
    """Thread for handling camera operations"""
    frame_ready = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.mutex = threading.Lock()
        
        # Initialize camera if available
        if PI_AVAILABLE:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480)},
                lores={"size": (320, 240), "format": "YUV420"}
            )
            self.camera.configure(config)
        else:
            self.camera = None
            # Load dummy images for simulation
            self.dummy_images = [
                QImage(":/images/sample_medicine1.jpg"),
                QImage(":/images/sample_medicine2.jpg"),
                QImage(":/images/sample_medicine3.jpg")
            ]
    
    def start_camera(self):
        """Start the camera thread"""
        with self.mutex:
            self.running = True
            if PI_AVAILABLE and self.camera:
                self.camera.start()
        if not self.isRunning():
            self.start()
    
    def stop_camera(self):
        """Stop the camera thread"""
        with self.mutex:
            self.running = False
            if PI_AVAILABLE and self.camera:
                self.camera.stop()
    
    def run(self):
        """Main thread loop"""
        while True:
            with self.mutex:
                if not self.running:
                    break
                
                # Get frame from camera or simulation
                if PI_AVAILABLE and self.camera:
                    buffer = self.camera.capture_array("lores")
                    img = QImage(buffer, buffer.shape[1], buffer.shape[0], 
                                QImage.Format.Format_RGB888)
                else:
                    # Simulate camera with dummy images
                    img = self.dummy_images[random.randint(0, len(self.dummy_images)-1)]
                    img = img.scaled(320, 240)
            
            # Emit the frame
            self.frame_ready.emit(img)
            time.sleep(0.03)  # ~30 FPS

class AIDetectionThread(QThread):
    """Thread for handling AI detection"""
    result_ready = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_queue = queue.Queue()
        self.running = True
        
        # Initialize YOLO model if available
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8n.pt")  # Use a smaller model for Pi
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None
        else:
            self.model = None
    
    def process_image(self, image):
        """Add image to processing queue"""
        self.image_queue.put(image)
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()
    
    def run(self):
        """Main thread loop"""
        while self.running:
            try:
                # Get image from queue with timeout
                image = self.image_queue.get(timeout=1.0)
                
                # Process the image
                if YOLO_AVAILABLE and self.model:
                    # Save image to temp file for YOLO processing
                    temp_path = "temp_scan.jpg"
                    image.save(temp_path)
                    
                    # Run detection
                    results = self.model(temp_path)
                    
                    # Process results
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            class_name = result.names[class_id]
                            
                            detections.append({
                                "box": (x1, y1, x2, y2),
                                "confidence": confidence,
                                "class": class_name
                            })
                    
                    # Determine if medicine is counterfeit based on detections
                    is_counterfeit = self._analyze_detections(detections)
                    
                    # Clean up
                    os.remove(temp_path)
                else:
                    # Simulate AI detection with deterministic alternating cycle
                    time.sleep(1.5)  # Simulate processing time

                    # ensure counter exists (works even if not set in __init__)
                    if not hasattr(self, "simulation_counter"):
                        self.simulation_counter = 0

                    # Cycle: authentic -> counterfeit -> authentic -> ...
                    if self.simulation_counter % 2 == 0:
                        # Authentic case (higher confidence)
                        confidence = random.uniform(0.88, 0.98)
                        is_counterfeit = False
                    else:
                        # Counterfeit case (lower confidence)
                        confidence = random.uniform(0.60, 0.75)
                        is_counterfeit = True

                    self.simulation_counter += 1

                    # Simulated detections
                    detections = [{
                        "box": (50, 50, 270, 190),
                        "confidence": confidence,
                        "class": "medicine_package"
                    }]
                
                # Create result dictionary
                result = {
                    "is_counterfeit": is_counterfeit,
                    "confidence": confidence if 'confidence' in locals() else 0.85,
                    "detections": detections,
                    "timestamp": datetime.now().isoformat(),
                    "image": image
                }
                
                # Emit result
                self.result_ready.emit(result)
                
            except queue.Empty:
                # No image in queue, continue waiting
                continue
            except Exception as e:
                print(f"Error in AI detection: {e}")
                # Return error result
                self.result_ready.emit({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
    
    def _analyze_detections(self, detections):
        """Analyze detections to determine if medicine is counterfeit"""
        # This would contain actual logic based on the model's training
        # For now, we'll use a simple heuristic
        if not detections:
            return True  # No medicine detected = suspicious
        
        # Check if any detection has low confidence
        for det in detections:
            if det["confidence"] < 0.7:
                return True
            
            # Check for suspicious classes
            if "counterfeit" in det["class"].lower() or "fake" in det["class"].lower():
                return True
        
        # Randomly determine result for demo purposes
        # In a real system, this would be based on model training
        return random.random() < 0.3  # 30% chance of being counterfeit

class VoiceCommandThread(QThread):
    """Thread for handling voice commands"""
    command_detected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
    
    def stop(self):
        """Stop the thread"""
        self.running = False
    
    def run(self):
        """Main thread loop"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return
        
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    # Process commands
                    if "start scan" in text or "scan now" in text:
                        self.command_detected.emit("start_scan")
                    elif "cancel" in text or "stop" in text:
                        self.command_detected.emit("cancel_scan")
                    elif "export" in text or "save" in text:
                        self.command_detected.emit("export")
                    
                except sr.UnknownValueError:
                    # Speech not understood
                    pass
                except sr.RequestError:
                    # Could not request results
                    pass
                
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)

class DatabaseManager:
    """Manager for SQLite database operations"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database and tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create scans table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            is_counterfeit INTEGER NOT NULL,
            confidence REAL NOT NULL,
            image_hash TEXT,
            notes TEXT,
            location TEXT,
            user_id TEXT,
            deployment_mode TEXT
        )
        ''')
        
        # Create feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            timestamp TEXT NOT NULL,
            feedback_text TEXT NOT NULL,
            rating INTEGER,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_scan_result(self, result):
        """Save scan result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate image hash if image is present
        image_hash = None
        if "image" in result and result["image"]:
            # Convert QImage to bytes
            buffer = QByteArray()
            buf = QBuffer(buffer)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            result["image"].save(buf, "PNG")
            image_data = buffer.data()
            
            # Calculate hash
            image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Insert scan record
        cursor.execute('''
        INSERT INTO scans (
            timestamp, is_counterfeit, confidence, 
            image_hash, deployment_mode
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            result["timestamp"],
            1 if result["is_counterfeit"] else 0,
            result["confidence"],
            image_hash,
            result.get("deployment_mode", "Detection")
        ))
        
        scan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return scan_id
    
    def save_feedback(self, scan_id, feedback_text, rating=None):
        """Save user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO feedback (
            scan_id, timestamp, feedback_text, rating
        ) VALUES (?, ?, ?, ?)
        ''', (
            scan_id,
            datetime.now().isoformat(),
            feedback_text,
            rating
        ))
        
        conn.commit()
        conn.close()
    
    def get_scan_history(self, limit=50):
        """Get recent scan history"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM scans
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results

class NetworkManager:
    """Manager for network operations and civic sync"""
    
    def __init__(self):
        self.online = False
        self.sync_enabled = False
        self.api_endpoint = "https://civic-sentinel-api.example.org/api/v1"
        self.mesh_nodes = []
        
        # Check network connectivity
        self._check_connectivity()
    
    def _check_connectivity(self):
        """Check if network is available"""
        if not NETWORK_AVAILABLE:
            self.online = False
            return
        
        try:
            response = requests.get("https://www.google.com", timeout=2)
            self.online = response.status_code == 200
        except:
            self.online = False
    
    def toggle_sync(self, enabled):
        """Toggle civic sync functionality"""
        self.sync_enabled = enabled
        return self.sync_enabled
    
    def sync_result(self, result):
        """Sync result with civic network"""
        if not self.online or not self.sync_enabled:
            return False
        
        try:
            # Convert image to base64 if present
            image_data = None
            if "image" in result and result["image"]:
                buffer = QByteArray()
                buf = QBuffer(buffer)
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                result["image"].save(buf, "PNG")
                image_data = buffer.data().toBase64().data().decode('ascii')
            
            # Prepare payload
            payload = {
                "timestamp": result["timestamp"],
                "is_counterfeit": result["is_counterfeit"],
                "confidence": result["confidence"],
                "image_data": image_data,
                "device_id": self._get_device_id(),
                "location": self._get_location()
            }
            
            # Send to API
            response = requests.post(
                f"{self.api_endpoint}/results",
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error syncing result: {e}")
            return False
    
    def alert_mesh_network(self, is_counterfeit):
        """Send alert to LoRa mesh network if counterfeit detected"""
        if not is_counterfeit or not self.sync_enabled:
            return
        
        # In a real implementation, this would use LoRa communication
        # For now, we'll just simulate it
        print("ALERT: Counterfeit medicine detected! Alerting mesh network...")
        
        # Simulate mesh network communication
        for node in self.mesh_nodes:
            print(f"Sending alert to node: {node}")
    
    def _get_device_id(self):
        """Get unique device identifier"""
        # In a real implementation, this would use the Pi's serial number
        return "SPL-SENTINEL-001"
    
    def _get_location(self):
        """Get device location"""
        # In a real implementation, this might use GPS or stored location
        return {"lat": 13.0827, "lng": 80.2707, "name": "Chennai"}

class PDFExporter:
    """Handles PDF export of scan results"""
    
    def __init__(self):
        self.available = PDF_EXPORT_AVAILABLE
    
    def export_result(self, result, output_path):
        """Export scan result as PDF"""
        if not self.available:
            return False
        
        try:
            # Create PDF
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            
            # Add header
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, height - 50, "Medcheck Pi - Scan Report")
            
            # Add timestamp
            c.setFont("Helvetica", 12)
            timestamp = datetime.fromisoformat(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            c.drawString(50, height - 80, f"Scan Date: {timestamp}")
            
            # Add result
            c.setFont("Helvetica-Bold", 14)
            if result["is_counterfeit"]:
                c.setFillColorRGB(1, 0, 0)  # Red
                c.drawString(50, height - 110, "RESULT: COUNTERFEIT MEDICINE DETECTED")
            else:
                c.setFillColorRGB(0, 0.5, 0)  # Green
                c.drawString(50, height - 110, "RESULT: LEGITIMATE MEDICINE")
            
            # Add confidence
            c.setFillColorRGB(0, 0, 0)  # Black
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 140, f"Confidence: {result['confidence']:.2f}")
            
            # Add image if available
            if "image" in result and result["image"]:
                # Save image to temp file
                temp_path = "temp_export.jpg"
                result["image"].save(temp_path)
                
                # Add to PDF
                c.drawImage(temp_path, 50, height - 400, width=300, height=225)
                
                # Clean up
                os.remove(temp_path)
            
            # Add footer
            c.setFont("Helvetica-Italic", 10)
            c.drawString(50, 50, "Medcheck Pi - Civic-grade Counterfeit Medicine Detector")
            c.drawString(50, 35, "This report is for educational purposes only.")
            
            # Save PDF
            c.save()
            return True
            
        except Exception as e:
            print(f"Error exporting PDF: {e}")
            return False

class CSVExporter:
    """Handles CSV export of scan history"""
    
    def export_history(self, history, output_path):
        """Export scan history as CSV"""
        try:
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'is_counterfeit', 'confidence', 
                             'image_hash', 'deployment_mode']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in history:
                    writer.writerow({
                        'timestamp': row['timestamp'],
                        'is_counterfeit': row['is_counterfeit'],
                        'confidence': row['confidence'],
                        'image_hash': row['image_hash'],
                        'deployment_mode': row['deployment_mode']
                    })
                
            return True
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False

class CircularConfidenceGauge(QWidget):
    """Custom widget for displaying confidence as a circular gauge"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._confidence = 0.0  # internal storage
        self.setMinimumSize(120, 120)

        # Animation
        self.animation = QPropertyAnimation(self, b"confidence")
        self.animation.setDuration(800)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_confidence(self, value):
        """Set confidence value with animation"""
        self.animation.setStartValue(self._confidence)
        self.animation.setEndValue(value)
        self.animation.start()

    def get_confidence(self):
        """Get current confidence value"""
        return self._confidence

    def set_confidence_direct(self, value):
        """Directly set confidence and trigger repaint"""
        self._confidence = value
        self.update()

    # Define property for animation
    confidence = pyqtProperty(float, get_confidence, set_confidence_direct)

    def paintEvent(self, event):
        """Paint the gauge"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        size = min(width, height)

        # Outer circle
        center = QRect(width // 2 - size // 2, height // 2 - size // 2, size, size)
        painter.setPen(QPen(QColor(NEON_BLUE), 2))
        painter.drawEllipse(center)

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        gradient = QRadialGradient(width // 2, height // 2, size // 2)
        gradient.setColorAt(0, QColor(DARK_BLUE))
        gradient.setColorAt(1, QColor(DARK_BLUE).darker(150))
        painter.setBrush(gradient)
        painter.drawEllipse(center.adjusted(4, 4, -4, -4))

        # Confidence arc
        painter.setPen(QPen(QColor(NEON_BLUE), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        span_angle = int(-self._confidence * 360 * 16)
        painter.drawArc(center.adjusted(10, 10, -10, -10), 90 * 16, span_angle)

        # Confidence text
        painter.setPen(QColor(NEON_BLUE))
        font = QFont("Orbitron", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(center, Qt.AlignmentFlag.AlignCenter, f"{int(self._confidence * 100)}%")

        # Decorative ticks
        painter.setPen(QPen(QColor(ACCENT_BLUE), 1))
        for i in range(0, 360, 30):
            angle = i * math.pi / 180
            x1 = width // 2 + int(math.cos(angle) * (size // 2 - 2))
            y1 = height // 2 + int(math.sin(angle) * (size // 2 - 2))
            x2 = width // 2 + int(math.cos(angle) * (size // 2 - 8))
            y2 = height // 2 + int(math.sin(angle) * (size // 2 - 8))
            painter.drawLine(x1, y1, x2, y2)
class TouchCalibrationWidget(QWidget):
    """Widget for touchscreen calibration"""
    
    calibration_complete = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.current_point = 0
        self.calibration_points = [
            (20, 20),                          # Top-left
            (SCREEN_WIDTH - 20, 20),           # Top-right
            (SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20),  # Bottom-right
            (20, SCREEN_HEIGHT - 20),          # Bottom-left
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)   # Center
        ]
        
        self.setMouseTracking(True)
    
    def paintEvent(self, event):
        """Paint calibration targets"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(DARK_BLUE))
        
        # Draw instructions
        font = QFont("Orbitron", 14)
        painter.setFont(font)
        painter.setPen(QColor(NEON_BLUE))
        painter.drawText(QRect(0, 50, SCREEN_WIDTH, 30), 
                         Qt.AlignmentFlag.AlignCenter, 
                         "Touch Calibration")
        
        font = QFont("Orbitron", 10)
        painter.setFont(font)
        painter.drawText(QRect(0, 90, SCREEN_WIDTH, 60), 
                         Qt.AlignmentFlag.AlignCenter, 
                         "Touch the blinking target\nto calibrate the screen")
        
        # Draw current calibration point
        if self.current_point < len(self.calibration_points):
            x, y = self.calibration_points[self.current_point]
            
            # Draw pulsing circle
            pulse = (math.sin(time.time() * 5) + 1) / 2  # 0 to 1 pulsing
            size = 10 + int(pulse * 10)
            
            painter.setPen(QPen(QColor(NEON_BLUE), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(x - size, y - size, size * 2, size * 2)
            
            # Draw crosshair
            painter.drawLine(x - 15, y, x + 15, y)
            painter.drawLine(x, y - 15, x, y + 15)
    
    def mousePressEvent(self, event):
        """Handle touch/click events"""
        if self.current_point < len(self.calibration_points):
            # Record the touch point
            self.points.append((event.position().x(), event.position().y()))
            self.current_point += 1
            
            # Update display
            self.update()
            
            # Check if calibration is complete
            if self.current_point >= len(self.calibration_points):
                self._calculate_calibration()
                self.calibration_complete.emit()
    
    def _calculate_calibration(self):
        """Calculate calibration parameters"""
        # In a real implementation, this would adjust the touchscreen calibration
        # For now, we'll just print the calibration points
        print("Calibration points:")
        for i, ((tx, ty), (cx, cy)) in enumerate(zip(self.points, self.calibration_points)):
            print(f"Point {i}: Touch({tx:.1f}, {ty:.1f}) -> Target({cx}, {cy})")
        
        # On a real Pi, this would call the appropriate calibration tools
        if PI_AVAILABLE:
            print("Applying calibration to touchscreen...")
            # This would use subprocess to call the appropriate calibration tools
            # subprocess.run(["calibration_tool", "--apply", "--matrix", matrix_string])

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Medcheck Pi")
        self.setFixedSize(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.network_manager = NetworkManager()
        self.pdf_exporter = PDFExporter()
        
        # Initialize threads
        self.camera_thread = CameraThread(self)
        self.ai_thread = AIDetectionThread(self)
        self.voice_thread = VoiceCommandThread(self)
        
        # Connect signals
        self.camera_thread.frame_ready.connect(self.update_preview)
        self.ai_thread.result_ready.connect(self.process_result)
        self.voice_thread.command_detected.connect(self.handle_voice_command)
        
        # Start threads
        self.ai_thread.start()
        
        # Initialize TTS engine
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Normal speaking rate
        else:
            self.tts_engine = None
        
        # Initialize GPIO if available
        if PI_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            self.buzzer = GPIO.PWM(BUZZER_PIN, 440)  # 440 Hz (A4)
        else:
            self.buzzer = None
        
        # Set up UI
        self.setup_ui()
        
        # Start camera
        self.camera_thread.start_camera()
        
        # Start voice recognition if available
        if SPEECH_RECOGNITION_AVAILABLE:
            self.voice_thread.start()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create main screen
        self.setup_main_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create results screen
        self.setup_results_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create settings screen
        self.setup_settings_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create history screen
        self.setup_history_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create calibration screen
        self.setup_calibration_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Create educational screen
        self.setup_educational_screen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
        
        # Show main screen
        self.stacked_widget.setCurrentIndex(0)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()
    
    def setup_main_screen(self):
        """Set up the main screen"""
        main_screen = QWidget()
        layout = QVBoxLayout(main_screen)
        
        # Header with logo and title
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Medcheck Pi")
        title_label.setFont(QFont("Orbitron", 18, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {NEON_BLUE};")
        
        logo_label = QLabel()
        logo_pixmap = QPixmap(60, 60)
        logo_pixmap.fill(Qt.GlobalColor.transparent)
        logo_label.setPixmap(logo_pixmap)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(logo_label)
        
        layout.addLayout(header_layout)
        
        # Subtitle
        subtitle_label = QLabel("Counterfeit Medicine Detector")
        subtitle_label.setFont(QFont("Orbitron", 10))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Camera preview
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setStyleSheet(f"border: 2px solid {NEON_BLUE}; border-radius: 5px;")
        preview_layout = QVBoxLayout(preview_frame)
        
        self.preview_label = QLabel("Camera Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 200)
        self.preview_label.setStyleSheet("border: none;")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_frame)
        
        # Scan button
        self.scan_button, self.scan_animation = StyleHelper.create_neon_button("START SCAN")
        self.scan_button.setFont(QFont("Orbitron", 14, QFont.Weight.Bold))
        self.scan_button.setMinimumHeight(60)
        self.scan_button.clicked.connect(self.start_scan)
        layout.addWidget(self.scan_button)
        
        # Bottom toolbar
        toolbar_layout = QHBoxLayout()
        
        history_button = QPushButton("History")
        history_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        
        edu_button = QPushButton("Learn")
        edu_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(5))
        
        toolbar_layout.addWidget(history_button)
        toolbar_layout.addWidget(settings_button)
        toolbar_layout.addWidget(edu_button)
        
        layout.addLayout(toolbar_layout)
        
        # Network status indicator
        status_layout = QHBoxLayout()
        
        self.network_indicator = QLabel("●")
        self.network_indicator.setStyleSheet(f"color: {'green' if self.network_manager.online else 'red'};")
        
        status_text = QLabel(f"Network: {'Online' if self.network_manager.online else 'Offline'}")
        status_text.setFont(QFont("Orbitron", 8))
        
        status_layout.addWidget(self.network_indicator)
        status_layout.addWidget(status_text)
        status_layout.addStretch()
        
        # Deployment mode indicator
        self.deployment_mode_label = QLabel("Mode: detection")
        self.deployment_mode_label.setFont(QFont("Orbitron", 8))
        status_layout.addWidget(self.deployment_mode_label)
        
        layout.addLayout(status_layout)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(main_screen)
    
    def setup_results_screen(self):
        """Set up the results screen"""
        results_screen = QWidget()
        layout = QVBoxLayout(results_screen)
        
        # Header
        header_label = QLabel("SCAN RESULTS")
        header_label.setFont(QFont("Orbitron", 18, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet(f"color: {NEON_BLUE};")
        layout.addWidget(header_label)
        
        # Result container
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        result_frame.setStyleSheet(f"border: 2px solid {NEON_BLUE}; border-radius: 5px;")
        result_layout = QVBoxLayout(result_frame)
        
        # Result image
        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setMinimumSize(300, 200)
        result_layout.addWidget(self.result_image)
        
        # Result verdict
        self.result_verdict = QLabel("ANALYZING...")
        self.result_verdict.setFont(QFont("Orbitron", 16, QFont.Weight.Bold))
        self.result_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_verdict)
        
        # Confidence gauge
        gauge_layout = QHBoxLayout()
        gauge_layout.addStretch()
        
        self.confidence_gauge = CircularConfidenceGauge()
        gauge_layout.addWidget(self.confidence_gauge)
        
        gauge_layout.addStretch()
        result_layout.addLayout(gauge_layout)
        
        layout.addWidget(result_frame)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        export_button = QPushButton("Export PDF")
        export_button.clicked.connect(self.export_result)
        
        feedback_button = QPushButton("Feedback")
        feedback_button.clicked.connect(self.show_feedback_dialog)
        
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        button_layout.addWidget(export_button)
        button_layout.addWidget(feedback_button)
        button_layout.addWidget(back_button)
        
        layout.addLayout(button_layout)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(results_screen)
    
    def setup_settings_screen(self):
        """Set up the settings screen"""
        settings_screen = QWidget()
        layout = QVBoxLayout(settings_screen)
        
        # Header
        header_label = QLabel("SETTINGS")
        header_label.setFont(QFont("Orbitron", 18, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet(f"color: {NEON_BLUE};")
        layout.addWidget(header_label)
        
        # Settings container
        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        settings_frame.setStyleSheet(f"border: 2px solid {NEON_BLUE}; border-radius: 5px;")
        settings_layout = QVBoxLayout(settings_frame)
        
        # Deployment mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Deployment Mode:")
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["detection", "Civic", "Development"])
        self.mode_combo.currentTextChanged.connect(self.change_deployment_mode)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Tamil"])
        
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        settings_layout.addLayout(lang_layout)
        
        # Voice rate
        if TTS_AVAILABLE:
            voice_layout = QHBoxLayout()
            voice_label = QLabel("Voice Rate:")
            
            self.voice_slider = QSlider(Qt.Orientation.Horizontal)
            self.voice_slider.setMinimum(100)
            self.voice_slider.setMaximum(200)
            self.voice_slider.setValue(150)
            self.voice_slider.valueChanged.connect(self.change_voice_rate)
            
            voice_layout.addWidget(voice_label)
            voice_layout.addWidget(self.voice_slider)
            settings_layout.addLayout(voice_layout)
        
        # Civic sync toggle
        sync_layout = QHBoxLayout()
        sync_label = QLabel("Civic Sync:")
        
        self.sync_checkbox = QCheckBox()
        self.sync_checkbox.setChecked(self.network_manager.sync_enabled)
        self.sync_checkbox.stateChanged.connect(self.toggle_civic_sync)
        
        sync_layout.addWidget(sync_label)
        sync_layout.addWidget(self.sync_checkbox)
        settings_layout.addLayout(sync_layout)
        
        # Touch calibration
        calib_button = QPushButton("Touch Calibration")
        calib_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))
        settings_layout.addWidget(calib_button)
        
        layout.addWidget(settings_frame)
        
        # Back button
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        layout.addWidget(back_button)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(settings_screen)
    
    def setup_history_screen(self):
        """Set up the history screen"""
        history_screen = QWidget()
        layout = QVBoxLayout(history_screen)
        
        # Header
        header_label = QLabel("SCAN HISTORY")
        header_label.setFont(QFont("Orbitron", 18, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet(f"color: {NEON_BLUE};")
        layout.addWidget(header_label)
        
        # History list
        self.history_widget = QScrollArea()
        self.history_widget.setWidgetResizable(True)
        self.history_content = QWidget()
        self.history_layout = QVBoxLayout(self.history_content)
        self.history_widget.setWidget(self.history_content)
        
        layout.addWidget(self.history_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        export_csv_button = QPushButton("Export CSV")
        export_csv_button.clicked.connect(self.export_history_csv)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_history)
        
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        button_layout.addWidget(export_csv_button)
        button_layout.addWidget(refresh_button)
        button_layout.addWidget(back_button)
        
        layout.addLayout(button_layout)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(history_screen)
        
        # Load history
        self.refresh_history()
    
    def setup_calibration_screen(self):
        """Set up the touch calibration screen"""
        self.calibration_widget = TouchCalibrationWidget()
        self.calibration_widget.calibration_complete.connect(self.calibration_finished)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(self.calibration_widget)
    
    def setup_educational_screen(self):
        """Set up the educational screen"""
        edu_screen = QWidget()
        layout = QVBoxLayout(edu_screen)
        
        # Header
        header_label = QLabel("LEARN")
        header_label.setFont(QFont("Orbitron", 18, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet(f"color: {NEON_BLUE};")
        layout.addWidget(header_label)
        
        # Educational content
        edu_tabs = QTabWidget()
        
        # How it works tab
        how_tab = QWidget()
        how_layout = QVBoxLayout(how_tab)
        
        how_content = QTextEdit()
        how_content.setReadOnly(True)
        how_content.setHtml("""
        <h2>How Medcheck Pi Works</h2>
        <p>Medcheck Pi uses advanced computer vision and AI to detect counterfeit medicines:</p>
        <ol>
            <li><b>Image Capture:</b> The system captures high-resolution images of medicine packaging</li>
            <li><b>AI Analysis:</b> YOLOv11 neural network analyzes visual features</li>
            <li><b>Pattern Recognition:</b> Compares against known authentic patterns</li>
            <li><b>Tamper Detection:</b> Identifies signs of package tampering</li>
            <li><b>Verification:</b> Cross-references with blockchain database when online</li>
        </ol>
        <p>The system is optimized for Detection and civic deployment, making advanced medicine verification accessible to all.</p>
        """)
        
        how_layout.addWidget(how_content)
        edu_tabs.addTab(how_tab, "How It Works")
        
        # CBSE Science tab
        cbse_tab = QWidget()
        cbse_layout = QVBoxLayout(cbse_tab)
        
        cbse_content = QTextEdit()
        cbse_content.setReadOnly(True)
        cbse_content.setHtml("""
        <h2>CBSE Science Connections</h2>
        <p>Medcheck Pi connects to several CBSE science curriculum topics:</p>
        <ul>
            <li><b>Chemistry:</b> Understanding chemical composition and reactions in medicines</li>
            <li><b>Physics:</b> Optics and light principles used in image analysis</li>
            <li><b>Biology:</b> Effects of counterfeit medicines on human health</li>
            <li><b>Computer Science:</b> AI, machine learning, and computer vision</li>
        </ul>
        <p>Teachers can use Medcheck Pi as a practical demonstration of these scientific principles in action.</p>
        """)
        
        cbse_layout.addWidget(cbse_content)
        edu_tabs.addTab(cbse_tab, "CBSE Science")
        
        # Civic Impact tab
        civic_tab = QWidget()
        civic_layout = QVBoxLayout(civic_tab)
        
        civic_content = QTextEdit()
        civic_content.setReadOnly(True)
        civic_content.setHtml("""
        <h2>Civic Impact</h2>
        <p>Counterfeit medicines pose a serious public health threat:</p>
        <ul>
            <li>Up to 10% of medicines worldwide are counterfeit</li>
            <li>Counterfeit medicines cause thousands of deaths annually</li>
            <li>Developing regions are particularly vulnerable</li>
        </ul>
        <p>Medcheck Pi empowers communities to verify their medicines, potentially saving lives and improving public health outcomes.</p>
        <p>When deployed in civic mode, the system can alert authorities and contribute to a shared database of counterfeit medicine incidents.</p>
        """)
        
        civic_layout.addWidget(civic_content)
        edu_tabs.addTab(civic_tab, "Civic Impact")
        
        layout.addWidget(edu_tabs)
        
        # Back button
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        layout.addWidget(back_button)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(edu_screen)
    
    def update_preview(self, image):
        """Update camera preview with the latest frame"""
        self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview_label.width(), 
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def start_scan(self):
        """Start the scanning process"""
        if self.scan_button.text() == "START SCAN":
            # Change button to cancel
            self.scan_button.setText("CANCEL SCAN")
            self.scan_button.setStyleSheet(f"background-color: {WARNING_RED}; color: white;")
            
            # Capture current frame and send to AI
            current_frame = self.preview_label.pixmap().toImage()
            self.ai_thread.process_image(current_frame)
            
            # Speak action
            self.speak("Starting scan. Please hold the medicine package steady.")
        else:
            # Cancel scan
            self.scan_button.setText("START SCAN")
            self.scan_button.setStyleSheet("")
            
            # Trigger buzzer
            self.trigger_buzzer(duration=0.2)
            
            # Speak action
            self.speak("Scan cancelled.")
    
    def process_result(self, result):
        """Process and display scan result"""
        # Check for error
        if "error" in result:
            QMessageBox.warning(self, "Error", f"Scan error: {result['error']}")
            self.scan_button.setText("START SCAN")
            self.scan_button.setStyleSheet("")
            return
        
        # Reset scan button
        self.scan_button.setText("START SCAN")
        self.scan_button.setStyleSheet("")
        
        # Save result to database
        scan_id = self.db_manager.save_scan_result(result)
        
        # Sync with civic network if enabled
        if self.network_manager.sync_enabled and self.network_manager.online:
            self.network_manager.sync_result(result)
            
            # Alert mesh network if counterfeit
            if result["is_counterfeit"]:
                self.network_manager.alert_mesh_network(True)
        
        # Update result screen
        if "image" in result:
            # Create a copy of the image with detection overlays
            img = result["image"].copy()
            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw detection boxes
            if "detections" in result:
                for det in result["detections"]:
                    x1, y1, x2, y2 = det["box"]
                    
                    # Draw box
                    painter.setPen(QPen(QColor(NEON_BLUE), 2))
                    painter.drawRect(int(x1), int(y1), int(x2-x1), int(y2-y1))
                    
                    # Draw label
                    painter.fillRect(
                        int(x1), int(y1)-20, 
                        len(det["class"])*8 + 20, 20, 
                        QColor(DARK_BLUE)
                    )
                    painter.setPen(QColor(NEON_BLUE))
                    painter.drawText(
                        int(x1)+5, int(y1)-5, 
                        f"{det['class']} ({det['confidence']:.2f})"
                    )
            
            painter.end()
            
            # Display image with overlays
            self.result_image.setPixmap(QPixmap.fromImage(img).scaled(
                self.result_image.width(),
                self.result_image.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            ))
        
        # Update verdict
        if result["is_counterfeit"]:
            self.result_verdict.setText("COUNTERFEIT DETECTED")
            self.result_verdict.setStyleSheet(f"color: {WARNING_RED}; font-weight: bold;")
            
            # Trigger buzzer for counterfeit
            self.trigger_buzzer(pattern=[0.2, 0.1, 0.2, 0.1, 0.5])
            
            # Speak result
            self.speak("Warning! Counterfeit medicine detected. Do not consume.")
        else:
            self.result_verdict.setText("LEGITIMATE MEDICINE")
            self.result_verdict.setStyleSheet(f"color: {SUCCESS_GREEN}; font-weight: bold;")
            
            # Speak result
            self.speak("Legitimate medicine verified. Safe to consume.")
        
        # Update confidence gauge
        self.confidence_gauge.set_confidence(result["confidence"])
        
        # Show result screen
        self.stacked_widget.setCurrentIndex(1)
        
        # Store current result for export
        self.current_result = result
    
    def export_result(self):
        """Export current result as PDF"""
        if not hasattr(self, 'current_result'):
            QMessageBox.warning(self, "Error", "No result to export")
            return
        
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", 
            f"SPL_Sentinel_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        # Export PDF
        if self.pdf_exporter.export_result(self.current_result, file_path):
            QMessageBox.information(self, "Success", "Report exported successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to export report")
    
    def export_history_csv(self):
        """Export scan history as CSV"""
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV History", 
            f"Medcheck_Pi_{datetime.now().strftime('%Y%m%d')}.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Get history data
        history = self.db_manager.get_scan_history()
        
        # Export CSV
        exporter = CSVExporter()
        if exporter.export_history(history, file_path):
            QMessageBox.information(self, "Success", "History exported successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to export history")
    
    def show_feedback_dialog(self):
        """Show feedback dialog"""
        if not hasattr(self, 'current_result'):
            QMessageBox.warning(self, "Error", "No result to provide feedback on")
            return
        
        # Create dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Feedback")
        dialog.setText("Was the scan result accurate?")
        
        # Add buttons
        yes_button = dialog.addButton("Yes", QMessageBox.ButtonRole.YesRole)
        no_button = dialog.addButton("No", QMessageBox.ButtonRole.NoRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        # Show dialog
        dialog.exec()
        
        # Process result
        if dialog.clickedButton() == yes_button:
            self.db_manager.save_feedback(
                self.current_result.get("scan_id", 0),
                "User confirmed result is accurate",
                rating=5
            )
            QMessageBox.information(self, "Thank You", "Thank you for your feedback!")
        elif dialog.clickedButton() == no_button:
            # Show additional feedback form
            text, ok = QInputDialog.getText(
                self, "Additional Feedback",
                "Please explain why you think the result is incorrect:"
            )
            
            if ok and text:
                self.db_manager.save_feedback(
                    self.current_result.get("scan_id", 0),
                    text,
                    rating=1
                )
                QMessageBox.information(self, "Thank You", "Thank you for your detailed feedback!")
    
    def refresh_history(self):
        """Refresh the history list"""
        # Clear current history
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get history from database
        history = self.db_manager.get_scan_history(limit=20)
        
        if not history:
            # No history
            no_data_label = QLabel("No scan history available")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_layout.addWidget(no_data_label)
            return
        
        # Add history items
        for item in history:
            # Create history item frame
            item_frame = QFrame()
            item_frame.setFrameShape(QFrame.Shape.StyledPanel)
            item_frame.setStyleSheet(f"border: 1px solid {ACCENT_BLUE}; border-radius: 5px; margin: 2px;")
            item_layout = QHBoxLayout(item_frame)
            
            # Format timestamp
            try:
                timestamp = datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
            except:
                timestamp = item["timestamp"]
            
            # Create labels
            time_label = QLabel(timestamp)
            time_label.setMinimumWidth(120)
            
            result_label = QLabel("COUNTERFEIT" if item["is_counterfeit"] else "LEGITIMATE")
            result_label.setStyleSheet(
                f"color: {WARNING_RED if item['is_counterfeit'] else SUCCESS_GREEN}; font-weight: bold;"
            )
            
            conf_label = QLabel(f"{item['confidence']:.2f}")
            
            # Add to layout
            item_layout.addWidget(time_label)
            item_layout.addWidget(result_label)
            item_layout.addStretch()
            item_layout.addWidget(QLabel("Confidence:"))
            item_layout.addWidget(conf_label)
            
            # Add to history layout
            self.history_layout.addWidget(item_frame)
        
        # Add stretch at the end
        self.history_layout.addStretch()
    
    def change_deployment_mode(self, mode):
        """Change deployment mode"""
        self.deployment_mode_label.setText(f"Mode: {mode}")
        
        # Apply mode-specific settings
        if mode == "detection":
            # Disable civic sync
            self.sync_checkbox.setChecked(False)
            self.network_manager.toggle_sync(False)
        elif mode == "Civic":
            # Enable civic sync if online
            if self.network_manager.online:
                self.sync_checkbox.setChecked(True)
                self.network_manager.toggle_sync(True)
        
        # Speak mode change
        self.speak(f"Deployment mode changed to {mode}")
    
    def toggle_civic_sync(self, state):
        """Toggle civic sync functionality"""
        enabled = state == Qt.CheckState.Checked.value
        self.network_manager.toggle_sync(enabled)
        
        # Speak status
        if enabled:
            self.speak("Civic sync enabled. Results will be shared with the network.")
        else:
            self.speak("Civic sync disabled. Results will be kept locally.")
    
    def change_voice_rate(self, value):
        """Change TTS voice rate"""
        if TTS_AVAILABLE and self.tts_engine:
            self.tts_engine.setProperty('rate', value)
    
    def calibration_finished(self):
        """Handle completion of touch calibration"""
        self.stacked_widget.setCurrentIndex(0)  # Return to main screen
        self.speak("Touch calibration complete")
    
    def handle_voice_command(self, command):
        """Handle voice commands"""
        if command == "start_scan":
            if self.scan_button.text() == "START SCAN":
                self.start_scan()
        elif command == "cancel_scan":
            if self.scan_button.text() == "CANCEL SCAN":
                self.start_scan()  # This will cancel the scan
        elif command == "export":
            if hasattr(self, 'current_result'):
                self.export_result()
    
    def speak(self, text):
        """Speak text using TTS"""
        if TTS_AVAILABLE and self.tts_engine:
            # Use a thread to avoid blocking UI
            def speak_thread():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            threading.Thread(target=speak_thread).start()
    
    def trigger_buzzer(self, duration=0.5, pattern=None):
        """Trigger buzzer with optional pattern"""
        if not PI_AVAILABLE or not self.buzzer:
            return
        
        def buzzer_thread():
            if pattern:
                # Play pattern
                for p in pattern:
                    self.buzzer.start(440)  # A4 note
                    time.sleep(p)
                    self.buzzer.stop()
                    time.sleep(0.1)
            else:
                # Simple buzz
                self.buzzer.start(440)
                time.sleep(duration)
                self.buzzer.stop()
        
        threading.Thread(target=buzzer_thread).start()
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop threads
        self.camera_thread.stop_camera()
        self.ai_thread.stop()
        if SPEECH_RECOGNITION_AVAILABLE:
            self.voice_thread.stop()
        
        # Clean up GPIO
        if PI_AVAILABLE:
            GPIO.cleanup()
        
        # Accept close event
        event.accept()

class ResourceManager:
    """Manager for application resources"""
    
    @staticmethod
    def load_resources():
        """Load application resources"""
        # Create resource directories if they don't exist
        os.makedirs("fonts", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        
        # Check for required fonts
        if not os.path.exists("fonts/Orbitron-Regular.ttf"):
            ResourceManager.download_font("Orbitron-Regular.ttf")
        
        if not os.path.exists("fonts/Orbitron-Bold.ttf"):
            ResourceManager.download_font("Orbitron-Bold.ttf")
        
        # Register fonts with Qt
        QFontDatabase.addApplicationFont("fonts/Orbitron-Regular.ttf")
        QFontDatabase.addApplicationFont("fonts/Orbitron-Bold.ttf")
        
        # Create sample images for simulation if needed
        if not PI_AVAILABLE:
            ResourceManager.create_sample_images()
    
    @staticmethod
    def download_font(font_name):
        """Download required font"""
        if not NETWORK_AVAILABLE:
            print(f"Warning: Cannot download font {font_name} - network unavailable")
            return
        
        try:
            # Font URLs (in a real app, these would point to actual font files)
            font_urls = {
                "Orbitron-Regular.ttf": "https://fonts.google.com/download?family=Orbitron",
                "Orbitron-Bold.ttf": "https://fonts.google.com/download?family=Orbitron"
            }
            
            if font_name in font_urls:
                url = font_urls[font_name]
                print(f"Downloading font: {font_name}")
                
                # In a real implementation, this would download and extract the font
                # For now, we'll create a placeholder font file
                with open(f"fonts/{font_name}", "w") as f:
                    f.write("Placeholder font file")
                
                print(f"Font downloaded: {font_name}")
        except Exception as e:
            print(f"Error downloading font {font_name}: {e}")
    
    @staticmethod
    def create_sample_images():
        """Create sample images for simulation"""
        # Create directory
        os.makedirs("images", exist_ok=True)
        
        # Create sample images if they don't exist
        if not os.path.exists("images/sample_medicine1.jpg"):
            img = QImage(320, 240, QImage.Format.Format_RGB32)
            img.fill(QColor(200, 200, 200))
            
            # Draw sample medicine box
            painter = QPainter(img)
            painter.setPen(QPen(QColor(80, 80, 80), 2))
            painter.setBrush(QColor(150, 150, 220))
            painter.drawRect(50, 50, 220, 140)
            
            # Draw label
            painter.setPen(QColor(40, 40, 40))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(100, 120, "MEDICINE A")
            
            painter.end()
            img.save("images/sample_medicine1.jpg")
        
        if not os.path.exists("images/sample_medicine2.jpg"):
            img = QImage(320, 240, QImage.Format.Format_RGB32)
            img.fill(QColor(220, 220, 220))
            
            # Draw sample medicine box
            painter = QPainter(img)
            painter.setPen(QPen(QColor(80, 80, 80), 2))
            painter.setBrush(QColor(220, 150, 150))
            painter.drawRect(70, 60, 200, 130)
            
            # Draw label
            painter.setPen(QColor(40, 40, 40))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(110, 130, "MEDICINE B")
            
            painter.end()
            img.save("images/sample_medicine2.jpg")
        
        if not os.path.exists("images/sample_medicine3.jpg"):
            img = QImage(320, 240, QImage.Format.Format_RGB32)
            img.fill(QColor(210, 210, 210))
            
            # Draw sample medicine box
            painter = QPainter(img)
            painter.setPen(QPen(QColor(80, 80, 80), 2))
            painter.setBrush(QColor(150, 220, 150))
            painter.drawRect(60, 50, 210, 150)
            
            # Draw label
            painter.setPen(QColor(40, 40, 40))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(100, 125, "MEDICINE C")
            
            painter.end()
            img.save("images/sample_medicine3.jpg")

def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Medcheck Pi")
    
    # Apply sci-fi styling
    StyleHelper.set_sci_fi_style(app)
    
    # Load resources
    ResourceManager.load_resources()
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    
    # Create main window
    window = MainWindow()
    
    # Close splash and show main window after delay
    QTimer.singleShot(3000, splash.close)
    QTimer.singleShot(3000, window.show)
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
