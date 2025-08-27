import os
import cv2
import time
import pickle
import threading
import numpy as np
import json
import sqlite3
from collections import defaultdict
from math import atan2, degrees, sqrt
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from sklearn.metrics.pairwise import cosine_similarity
import sys

# sklearn & joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import joblib

# mediapipe & tts
import mediapipe as mp
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

# TensorFlow for embeddings and improved training
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not available. Install with: pip install tensorflow")

# ArcFace face recognition
try:
    import insightface
    from insightface.app import FaceAnalysis
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    print("Warning: insightface not available. Install with: pip install insightface onnxruntime")

DATA_FILE = "palm_features.pkl"
MODEL_FILE = "palm_model.joblib"
EMBEDDINGS_FILE = "palm_embeddings.npy"
IMAGES_FILE = "palm_images.npy"
LABELS_FILE = "palm_labels.npy"
DB_FILE = "biometric_auth.db"

class DatabaseManager:
    """Manages SQLite database for storing user face embeddings"""
    
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    face_embedding TEXT NOT NULL,
                    palm_features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"Database initialized: {self.db_file}")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def add_user(self, name, face_embedding, palm_features=None):
        """Add a new user with face embedding and optional palm features"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Convert numpy arrays to JSON strings
            face_emb_json = json.dumps(face_embedding.tolist())
            palm_features_json = json.dumps(palm_features.tolist()) if palm_features is not None else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO users (name, face_embedding, palm_features)
                VALUES (?, ?, ?)
            ''', (name, face_emb_json, palm_features_json))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
    
    def get_user_face_embedding(self, name):
        """Get face embedding for a specific user"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('SELECT face_embedding FROM users WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return np.array(json.loads(result[0]), dtype=np.float32)
            return None
        except Exception as e:
            print(f"Error getting user embedding: {e}")
            return None
    
    def get_user_palm_features(self, name):
        """Get palm features for a specific user"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('SELECT palm_features FROM users WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result and result[0]:
                return np.array(json.loads(result[0]), dtype=np.float32)
            return None
        except Exception as e:
            print(f"Error getting palm features: {e}")
            return None
    
    def get_all_users(self):
        """Get list of all registered users"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name FROM users')
            results = cursor.fetchall()
            
            conn.close()
            
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error getting users: {e}")
            return []
    
    def delete_user(self, name):
        """Delete a user from the database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM users WHERE name = ?', (name,))
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False

class FaceRecognitionSystem:
    """Handles face detection and recognition with multiple fallback methods"""
    
    def __init__(self):
        self.app = None
        self.face_cascade = None
        self.init_face_detection()
        
    def init_face_detection(self):
        """Initialize face detection systems with fallbacks"""
        # Try to initialize ArcFace first
        if ARCFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(name='buffalo_l')
                self.app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU (-1) for better compatibility
                print("ArcFace initialized successfully")
                return
            except Exception as e:
                print(f"Failed to initialize ArcFace: {e}")
                self.app = None
        
        # Initialize OpenCV Haar cascade as fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    self.face_cascade = None
                    print("Failed to load Haar cascade")
                else:
                    print("OpenCV Haar cascade initialized as fallback")
            else:
                print("Haar cascade file not found")
        except Exception as e:
            print(f"Failed to initialize Haar cascade: {e}")
    
    def extract_face_embedding(self, image):
        """Extract face embedding with multiple fallback methods"""
        if image is None or image.size == 0:
            print("Invalid image provided")
            return None
            
        try:
            # Convert image format if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Image is in BGR format, convert to RGB for ArcFace
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Method 1: Try ArcFace
            if self.app is not None:
                embedding = self._extract_arcface_embedding(rgb_image)
                if embedding is not None:
                    return embedding
            
            # Method 2: Fallback to OpenCV + simple embedding
            return self._extract_opencv_embedding(rgb_image)
            
        except Exception as e:
            print(f"Face embedding extraction failed: {e}")
            return None
    
    def _extract_arcface_embedding(self, rgb_image):
        """Extract embedding using ArcFace"""
        try:
            faces = self.app.get(rgb_image)
            if len(faces) == 0:
                print("No faces detected by ArcFace")
                return None
            
            # Get the face with highest confidence
            best_face = max(faces, key=lambda x: x.det_score)
            print(f"ArcFace detected face with confidence: {best_face.det_score:.3f}")
            
            # Ensure embedding is normalized
            embedding = best_face.embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"ArcFace embedding extraction failed: {e}")
            return None
    
    def _extract_opencv_embedding(self, rgb_image):
        """Fallback embedding extraction using OpenCV"""
        try:
            # Convert to grayscale
            if len(rgb_image.shape) == 3:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = rgb_image
            
            # Detect faces
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(300, 300)
                )
            else:
                # Very basic face detection using image center
                h, w = gray.shape
                faces = [(w//4, h//4, w//2, h//2)]  # Center rectangle as "face"
                print("Using center region as face (no cascade available)")
            
            if len(faces) == 0:
                print("No faces detected by OpenCV")
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region with padding
            padding = max(10, min(w, h) // 10)
            y1 = max(0, y - padding)
            y2 = min(gray.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(gray.shape[1], x + w + padding)
            
            face_region = gray[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Create stable embedding
            face_resized = cv2.resize(face_region, (64, 64))
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Create embedding based on image statistics and simple features
            embedding = np.zeros(512, dtype=np.float32)
            
            # Basic statistics
            embedding[0] = np.mean(face_normalized)
            embedding[1] = np.std(face_normalized)
            embedding[2] = np.min(face_normalized)
            embedding[3] = np.max(face_normalized)
            
            # Histogram features
            hist = cv2.calcHist([face_resized], [0], None, [16], [0, 256])
            hist_norm = hist.flatten() / (hist.sum() + 1e-8)
            embedding[4:20] = hist_norm
            
            # Gradient features
            grad_x = cv2.Sobel(face_normalized, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_normalized, cv2.CV_32F, 0, 1, ksize=3)
            embedding[20] = np.mean(np.abs(grad_x))
            embedding[21] = np.mean(np.abs(grad_y))
            
            # LBP-like features (simplified)
            rows, cols = face_normalized.shape
            if rows > 2 and cols > 2:
                center = face_normalized[1:rows-1, 1:cols-1]
                neighbors = [
                    face_normalized[0:rows-2, 0:cols-2],  # top-left
                    face_normalized[0:rows-2, 1:cols-1],  # top
                    face_normalized[0:rows-2, 2:cols],    # top-right
                    face_normalized[1:rows-1, 2:cols],    # right
                    face_normalized[2:rows, 2:cols],      # bottom-right
                    face_normalized[2:rows, 1:cols-1],    # bottom
                    face_normalized[2:rows, 0:cols-2],    # bottom-left
                    face_normalized[1:rows-1, 0:cols-2],  # left
                ]
                
                for i, neighbor in enumerate(neighbors[:8]):
                    if i + 22 < len(embedding):
                        embedding[22 + i] = np.mean(neighbor > center)
            
            # Fill remaining with consistent pseudo-random features based on image content
            seed_value = int(np.sum(face_normalized * 1000)) % (2**31)
            np.random.seed(seed_value)
            remaining_features = np.random.normal(0, 0.1, 512 - 30).astype(np.float32)
            embedding[30:] = remaining_features
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            print(f"OpenCV face detected - size: {w}x{h}, position: ({x},{y})")
            return embedding
            
        except Exception as e:
            print(f"OpenCV embedding extraction failed: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.6):
        """Compare two face embeddings using cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        
        try:
            # Ensure embeddings are numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # Normalize embeddings
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2)
            
            # Ensure similarity is in valid range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return similarity > threshold, float(similarity)
            
        except Exception as e:
            print(f"Embedding comparison failed: {e}")
            return False, 0.0
    
    def test_face_detection(self):
        """Test face detection with camera feed"""
        print("Testing face detection...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera for testing")
            return False
        
        print("Camera opened. Look at the camera to test face detection.")
        print("Press 'q' to quit, 's' to save test image")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Test face detection every few frames to reduce processing load
            if frame_count % 5 == 0:
                embedding = self.extract_face_embedding(frame)
                
                if embedding is not None:
                    detection_count += 1
                    cv2.putText(frame, "FACE DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"Embedding shape: {embedding.shape}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detection rate: {detection_count}/{frame_count//5}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw detection box
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, "Position your face in center, ensure good lighting", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f"Detection rate: {detection_count}/{frame_count//5}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save test image", 
                       (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Face Detection Test", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save test image
                test_filename = f"face_test_{int(time.time())}.jpg"
                cv2.imwrite(test_filename, frame)
                print(f"Test image saved as: {test_filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Face detection test completed. Detection rate: {detection_count}/{frame_count//5}")
        return True

class BiometricAuthenticationSystem:
    """Main system combining face and palm recognition"""
    
    def __init__(self):
        # Initialize subsystems
        self.face_system = FaceRecognitionSystem()
        self.palm_system = PalmBiometricSystem()
        self.db_manager = DatabaseManager()
        
        # Authentication state
        self.current_user = None
        self.face_authenticated = False
        self.palm_authenticated = False
        
        print("Biometric Authentication System initialized")
        
    def register_new_user(self, name, num_face_images=5, num_palm_scans=5):
        """Register a new user with both face and palm biometrics"""
        if not name or not name.strip():
            return False, "Name is required"
        
        name = name.strip()
        
        # Check if user already exists
        existing_users = self.db_manager.get_all_users()
        if name in existing_users:
            return False, f"User '{name}' already exists"
        
        print(f"Starting registration for user: {name}")
        speak(f"Starting registration for {name}")
        
        # Step 1: Face Registration
        print("Step 1: Face Registration")
        speak("Step 1: Face registration. Look at the camera.")
        
        face_embeddings = []
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        face_captured = 0
        last_capture_time = 0
        consecutive_detections = 0
        
        start_time = time.time()
        timeout = 60  # 60 seconds timeout
        
        while face_captured < num_face_images and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Try to extract face embedding
            embedding = self.face_system.extract_face_embedding(frame)
            
            if embedding is not None:
                consecutive_detections += 1
                
                # Draw face detection box and instructions
                cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 255, 0), 2)
                cv2.putText(frame, f"Face detected - {face_captured}/{num_face_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Stable detection: {consecutive_detections}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Capture when face is stable for a few frames
                current_time = time.time()
                if consecutive_detections >= 10 and current_time - last_capture_time >= 2:
                    face_embeddings.append(embedding)
                    face_captured += 1
                    last_capture_time = current_time
                    consecutive_detections = 0
                    speak(f"Face captured {face_captured}")
                    print(f"Face captured {face_captured}/{num_face_images}")
                    
                    # Visual feedback
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
                    cv2.imshow(f"Face Registration - {name}", frame)
                    cv2.waitKey(800)  # Show capture feedback
            else:
                consecutive_detections = 0
                cv2.putText(frame, "No face detected - Position face in center", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Ensure good lighting and face clearly visible", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show progress and instructions
            remaining_time = int(timeout - (time.time() - start_time))
            cv2.putText(frame, f"Time remaining: {remaining_time}s", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Face Registration - {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_embeddings) < 2:
            return False, f"Insufficient face captures: {len(face_embeddings)}. Need at least 2."
        
        # Calculate average face embedding
        avg_face_embedding = np.mean(face_embeddings, axis=0).astype(np.float32)
        
        # Step 2: Palm Registration (simplified for now)
        print("Step 2: Palm Registration")
        speak("Step 2: Palm registration. Show your right palm.")
        
        # For now, create a simple palm feature vector
        # In a full implementation, you would use the palm system
        palm_features = np.random.random(50).astype(np.float32)  # Placeholder
        
        # Store in database
        success = self.db_manager.add_user(name, avg_face_embedding, palm_features)
        if not success:
            return False, "Failed to save user data"
        
        speak(f"Registration complete for {name}")
        print(f"User {name} registered successfully with {len(face_embeddings)} face samples")
        return True, f"User {name} registered successfully"
    
    def authenticate_user(self, name):
        """Authenticate user using face recognition"""
        if not name or not name.strip():
            return False, "Name is required"
        
        name = name.strip()
        
        # Check if user exists
        stored_face_embedding = self.db_manager.get_user_face_embedding(name)
        if stored_face_embedding is None:
            return False, f"User '{name}' not found"
        
        print(f"Starting authentication for user: {name}")
        speak(f"Starting authentication for {name}")
        
        # Face Authentication
        print("Face Authentication")
        speak("Face authentication. Look at the camera.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        authenticated = False
        start_time = time.time()
        timeout = 30
        
        best_similarity = 0.0
        consecutive_good_matches = 0
        required_matches = 5
        
        while not authenticated and (time.time() - start_time) < timeout:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract face embedding
            current_embedding = self.face_system.extract_face_embedding(frame)
            
            if current_embedding is not None:
                # Compare with stored embedding
                is_match, similarity = self.face_system.compare_embeddings(
                    current_embedding, stored_face_embedding, threshold=0.5
                )
                
                best_similarity = max(best_similarity, similarity)
                
                if is_match:
                    consecutive_good_matches += 1
                    cv2.putText(frame, f"Match! Similarity: {similarity:.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confirming... {consecutive_good_matches}/{required_matches}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if consecutive_good_matches >= required_matches:
                        authenticated = True
                        speak("Authentication successful")
                        print(f"Face authenticated with similarity: {similarity:.3f}")
                        
                        # Show success feedback
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                        cv2.putText(frame, "AUTHENTICATED!", 
                                   (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.imshow(f"Face Authentication - {name}", frame)
                        cv2.waitKey(2000)
                        break
                else:
                    consecutive_good_matches = 0
                    cv2.putText(frame, f"No match. Similarity: {similarity:.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Best so far: {best_similarity:.3f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                consecutive_good_matches = 0
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Position face in center, ensure good lighting", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show remaining time
            remaining_time = int(timeout - (time.time() - start_time))
            cv2.putText(frame, f"Time remaining: {remaining_time}s", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Face Authentication - {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if authenticated:
            return True, f"User {name} authenticated successfully"
        else:
            return False, f"Authentication failed. Best similarity: {best_similarity:.3f}"
    
    def list_users(self):
        """Get list of all registered users"""
        return self.db_manager.get_all_users()
    
    def delete_user(self, name):
        """Delete a user from the system"""
        success = self.db_manager.delete_user(name)
        
        if success:
            speak(f"User {name} deleted successfully")
            return True, f"User {name} deleted successfully"
        else:
            return False, f"Failed to delete user {name}"


def speak(text):
    """Voice assistant via pyttsx3 (non-blocking)"""
    print(text)  # Always print as fallback
    
    if not TTS_AVAILABLE:
        return
        
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    
    # Run in separate thread to avoid blocking
    threading.Thread(target=_speak, daemon=True).start()


class PalmBiometricSystem:
    """Simplified palm biometric system for compatibility"""
    
    def __init__(self):
        try:
            # MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("MediaPipe hands initialized successfully")
        except Exception as e:
            print(f"Failed to initialize MediaPipe hands: {e}")
            self.hands = None
            self.mp_drawing = None
        
        # Data storage
        self.features = []
        self.labels = []
        self.model = None
        
        # Flags
        self.recognize_running = False
    
    def is_right_hand(self, landmarks):
        """Simple right hand detection"""
        if landmarks is None:
            return False
        
        try:
            # Use thumb position relative to other fingers
            thumb_tip = landmarks.landmark[4]
            index_tip = landmarks.landmark[8]
            pinky_tip = landmarks.landmark[20]
            
            # Right hand usually has thumb on left side when palm faces camera
            return thumb_tip.x < index_tip.x and thumb_tip.x < pinky_tip.x
        except:
            return True  # Default to True if detection fails
    
    def extract_features(self, landmarks, image=None):
        """Extract basic features from palm landmarks"""
        try:
            if landmarks is None:
                return np.zeros(50, dtype=np.float32)
            
            # Convert landmarks to numpy array
            features = []
            for landmark in landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad or truncate to fixed size
            features = np.array(features, dtype=np.float32)
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            else:
                features = features[:50]
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def scan_palm_for_user(self, user_id, num_scans=5):
        """Simplified palm scanning"""
        if self.hands is None:
            print("MediaPipe not available for palm scanning")
            return 0
        
        # For now, just return success without actual scanning
        # In a full implementation, this would capture palm data
        print(f"Palm scanning for {user_id} (simplified)")
        return num_scans
    
    def train_model(self):
        """Simplified model training"""
        print("Palm model training (simplified)")
        return True
    
    def delete_user(self, user_id):
        """Delete palm data for user"""
        print(f"Deleting palm data for {user_id}")
        return True


class BiometricApp:
    """Enhanced GUI for the biometric authentication system"""
    
    def __init__(self, root):
        self.root = root
        root.title("Enhanced Biometric Authentication System")
        root.geometry("600x700")
        root.resizable(True, True)
        
        # Initialize the main system
        try:
            self.auth_system = BiometricAuthenticationSystem()
            self.system_ready = True
        except Exception as e:
            print(f"Failed to initialize authentication system: {e}")
            self.system_ready = False
        
        # Create main frame with scrollbar
        self.setup_gui()
        
        # Initialize users list
        self.refresh_users()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_container, text="Enhanced Biometric Authentication", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # System status
        status_frame = ttk.LabelFrame(main_container, text="System Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        system_status = "Ready" if self.system_ready else "Error - Check Dependencies"
        status_color = "green" if self.system_ready else "red"
        
        self.system_status_label = ttk.Label(status_frame, text=f"System: {system_status}", 
                                           foreground=status_color)
        self.system_status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Display available features
        features_text = []
        if hasattr(self, 'auth_system') and self.auth_system:
            if self.auth_system.face_system.app is not None:
                features_text.append("ArcFace: Available")
            elif self.auth_system.face_system.face_cascade is not None:
                features_text.append("OpenCV Haar: Available")
            else:
                features_text.append("Face Detection: Basic fallback")
        
        features_label = ttk.Label(status_frame, text=" | ".join(features_text))
        features_label.grid(row=1, column=0, sticky=tk.W)
        
        # User Management Section
        user_frame = ttk.LabelFrame(main_container, text="User Management", padding="10")
        user_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # User selection
        ttk.Label(user_frame, text="Select User:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, state="readonly", width=20)
        self.user_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 10), pady=5)
        
        ttk.Button(user_frame, text="Refresh", command=self.refresh_users).grid(row=0, column=2, pady=5)
        
        # Main action buttons
        action_frame = ttk.LabelFrame(main_container, text="Main Operations", padding="10")
        action_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Button grid
        btn_register = ttk.Button(action_frame, text="Register New User", 
                                 command=self.register_user, width=20)
        btn_register.grid(row=0, column=0, padx=5, pady=5)
        
        btn_authenticate = ttk.Button(action_frame, text="Authenticate User", 
                                     command=self.authenticate_user, width=20)
        btn_authenticate.grid(row=0, column=1, padx=5, pady=5)
        
        btn_delete = ttk.Button(action_frame, text="Delete User", 
                               command=self.delete_user, width=20)
        btn_delete.grid(row=0, column=2, padx=5, pady=5)
        
        # Testing section
        test_frame = ttk.LabelFrame(main_container, text="Testing & Diagnostics", padding="10")
        test_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        btn_test_face = ttk.Button(test_frame, text="Test Face Detection", 
                                  command=self.test_face_detection, width=20)
        btn_test_face.grid(row=0, column=0, padx=5, pady=5)
        
        btn_test_camera = ttk.Button(test_frame, text="Test Camera", 
                                    command=self.test_camera, width=20)
        btn_test_camera.grid(row=0, column=1, padx=5, pady=5)
        
        btn_system_info = ttk.Button(test_frame, text="System Info", 
                                    command=self.show_system_info, width=20)
        btn_system_info.grid(row=0, column=2, padx=5, pady=5)
        
        # Status and progress
        status_container = ttk.Frame(main_container)
        status_container.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an operation")
        self.status_label = ttk.Label(status_container, textvariable=self.status_var, 
                                     foreground="blue", font=('Arial', 10))
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_container, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Exit button
        btn_exit = ttk.Button(main_container, text="Exit Application", 
                             command=self.quit_app, width=25)
        btn_exit.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        status_container.columnconfigure(0, weight=1)

    def set_status(self, text, progress=None):
        """Update status text and optionally progress"""
        self.status_var.set(text)
        if progress is not None:
            self.progress_var.set(progress)
        print(text)
        self.root.update_idletasks()

    def refresh_users(self):
        """Refresh the list of registered users"""
        if not self.system_ready:
            self.set_status("System not ready - cannot refresh users")
            return
        
        try:
            users = self.auth_system.list_users()
            self.user_combo['values'] = users
            if users:
                self.user_combo.set(users[0])
            else:
                self.user_combo.set("")
            self.set_status(f"Found {len(users)} registered users")
        except Exception as e:
            self.set_status(f"Error refreshing users: {e}")

    def register_user(self):
        """Register a new user"""
        if not self.system_ready:
            messagebox.showerror("Error", "System not ready. Check dependencies.")
            return
        
        name = simpledialog.askstring("User Registration", 
                                     "Enter user name (unique):\n\nTips for best results:\n"
                                     "- Use good lighting\n"
                                     "- Look directly at camera\n"
                                     "- Keep face steady", parent=self.root)
        if not name:
            return
        
        if messagebox.askquestion("Confirm Registration", 
                                 f"Register user '{name}'?\n\n"
                                 "This will capture multiple face images.\n"
                                 "Make sure you have good lighting and camera access.") != 'yes':
            return
        
        self.set_status(f"Starting registration for {name}...", 10)
        
        def _do_register():
            try:
                success, message = self.auth_system.register_new_user(name)
                self.root.after(0, lambda: self._registration_complete(success, message, name))
            except Exception as e:
                error_msg = f"Registration error: {str(e)}"
                self.root.after(0, lambda: self._registration_complete(False, error_msg, name))
        
        threading.Thread(target=_do_register, daemon=True).start()
    
    def _registration_complete(self, success, message, name):
        """Handle registration completion"""
        if success:
            self.set_status(f"Registration successful: {message}", 100)
            self.refresh_users()
            messagebox.showinfo("Success", f"User '{name}' registered successfully!")
        else:
            self.set_status(f"Registration failed: {message}", 0)
            messagebox.showerror("Registration Failed", message)
        
        # Reset progress after a delay
        self.root.after(3000, lambda: self.set_status("Ready", 0))

    def authenticate_user(self):
        """Authenticate selected user"""
        if not self.system_ready:
            messagebox.showerror("Error", "System not ready. Check dependencies.")
            return
        
        selected_user = self.user_var.get()
        if not selected_user:
            messagebox.showwarning("No User Selected", "Please select a user to authenticate.")
            return
        
        self.set_status(f"Starting authentication for {selected_user}...", 10)
        
        def _do_authenticate():
            try:
                success, message = self.auth_system.authenticate_user(selected_user)
                self.root.after(0, lambda: self._authentication_complete(success, message, selected_user))
            except Exception as e:
                error_msg = f"Authentication error: {str(e)}"
                self.root.after(0, lambda: self._authentication_complete(False, error_msg, selected_user))
        
        threading.Thread(target=_do_authenticate, daemon=True).start()
    
    def _authentication_complete(self, success, message, user):
        """Handle authentication completion"""
        if success:
            self.set_status(f"Authentication successful: {message}", 100)
            messagebox.showinfo("Access Granted", f"Welcome {user}!\n\n{message}")
        else:
            self.set_status(f"Authentication failed: {message}", 0)
            messagebox.showerror("Access Denied", f"Authentication failed for {user}:\n\n{message}")
        
        # Reset progress after a delay
        self.root.after(3000, lambda: self.set_status("Ready", 0))

    def delete_user(self):
        """Delete selected user"""
        selected_user = self.user_var.get()
        if not selected_user:
            messagebox.showwarning("No User Selected", "Please select a user to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete user '{selected_user}'?\n\n"
                              "This action cannot be undone."):
            try:
                success, message = self.auth_system.delete_user(selected_user)
                if success:
                    self.set_status(f"User deleted: {message}")
                    self.refresh_users()
                    messagebox.showinfo("Success", message)
                else:
                    self.set_status(f"Delete failed: {message}")
                    messagebox.showerror("Error", message)
            except Exception as e:
                error_msg = f"Delete error: {str(e)}"
                self.set_status(error_msg)
                messagebox.showerror("Error", error_msg)

    def test_face_detection(self):
        """Test face detection system"""
        if not self.system_ready:
            messagebox.showerror("Error", "System not ready. Check dependencies.")
            return
        
        self.set_status("Starting face detection test...")
        messagebox.showinfo("Face Detection Test", 
                           "Face detection test will start.\n\n"
                           "- Look at the camera\n"
                           "- Move your head slightly\n"
                           "- Press 'q' to quit\n"
                           "- Press 's' to save test image")
        
        def _test():
            try:
                self.auth_system.face_system.test_face_detection()
                self.root.after(0, lambda: self.set_status("Face detection test completed"))
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"Face detection test error: {e}"))
        
        threading.Thread(target=_test, daemon=True).start()

    def test_camera(self):
        """Test camera access"""
        self.set_status("Testing camera access...")
        
        def _test_camera():
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self.root.after(0, lambda: self.set_status("Camera test: SUCCESS"))
                        self.root.after(0, lambda: messagebox.showinfo("Camera Test", "Camera is working correctly!"))
                    else:
                        self.root.after(0, lambda: self.set_status("Camera test: Failed to read frame"))
                        self.root.after(0, lambda: messagebox.showerror("Camera Test", "Camera opened but failed to read frame"))
                else:
                    self.root.after(0, lambda: self.set_status("Camera test: Cannot open camera"))
                    self.root.after(0, lambda: messagebox.showerror("Camera Test", "Cannot open camera. Check permissions and connections."))
            except Exception as e:
                self.root.after(0, lambda: self.set_status(f"Camera test error: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Camera Test", f"Camera test failed: {e}"))
        
        threading.Thread(target=_test_camera, daemon=True).start()

    def show_system_info(self):
        """Display detailed system information"""
        info_parts = ["=== BIOMETRIC AUTHENTICATION SYSTEM INFO ===\n"]
        
        # System status
        info_parts.append(f"System Ready: {'Yes' if self.system_ready else 'No'}")
        
        if self.system_ready:
            # Face detection capabilities
            if hasattr(self.auth_system.face_system, 'app') and self.auth_system.face_system.app:
                info_parts.append("Face Detection: ArcFace (High Accuracy)")
            elif hasattr(self.auth_system.face_system, 'face_cascade') and self.auth_system.face_system.face_cascade:
                info_parts.append("Face Detection: OpenCV Haar Cascade (Medium Accuracy)")
            else:
                info_parts.append("Face Detection: Basic Fallback (Low Accuracy)")
        
        # Database info
        try:
            users = self.auth_system.list_users() if self.system_ready else []
            info_parts.append(f"Registered Users: {len(users)}")
            if users:
                info_parts.append(f"Users: {', '.join(users)}")
        except:
            info_parts.append("Registered Users: Error reading database")
        
        # Dependencies status
        info_parts.append("\n=== DEPENDENCIES STATUS ===")
        info_parts.append(f"OpenCV: {'Available' if 'cv2' in sys.modules else 'Missing'}")
        info_parts.append(f"MediaPipe: {'Available' if 'mediapipe' in sys.modules else 'Missing'}")
        info_parts.append(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Missing'}")
        info_parts.append(f"InsightFace: {'Available' if ARCFACE_AVAILABLE else 'Missing'}")
        info_parts.append(f"Text-to-Speech: {'Available' if TTS_AVAILABLE else 'Missing'}")
        
        # Camera status
        info_parts.append("\n=== CAMERA STATUS ===")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    info_parts.append("Camera: Working correctly")
                    info_parts.append(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    info_parts.append("Camera: Opens but cannot read frames")
            else:
                info_parts.append("Camera: Cannot open (check permissions/connections)")
        except Exception as e:
            info_parts.append(f"Camera: Error - {e}")
        
        # Recommendations
        info_parts.append("\n=== RECOMMENDATIONS ===")
        if not ARCFACE_AVAILABLE:
            info_parts.append("• Install insightface for better face recognition accuracy")
            info_parts.append("  pip install insightface onnxruntime")
        if not TF_AVAILABLE:
            info_parts.append("• Install tensorflow for advanced features")
            info_parts.append("  pip install tensorflow")
        if not TTS_AVAILABLE:
            info_parts.append("• Install pyttsx3 for voice feedback")
            info_parts.append("  pip install pyttsx3")
        
        info_parts.append("\n=== TROUBLESHOOTING TIPS ===")
        info_parts.append("• Ensure good lighting for face detection")
        info_parts.append("• Position face in center of camera view")
        info_parts.append("• Keep face steady during capture/authentication")
        info_parts.append("• Check camera permissions in system settings")
        info_parts.append("• Try different lighting conditions if detection fails")
        
        info_text = "\n".join(info_parts)
        
        # Create scrollable dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("System Information")
        dialog.geometry("600x500")
        dialog.resizable(True, True)
        
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def quit_app(self):
        """Clean shutdown of the application"""
        if hasattr(self, 'auth_system') and hasattr(self.auth_system, 'palm_system'):
            self.auth_system.palm_system.recognize_running = False
        
        self.set_status("Shutting down...")
        self.root.after(500, self.root.quit)


def main():
    """Main application entry point"""
    print("Starting Enhanced Biometric Authentication System...")
    
    # Check basic requirements
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        print(f"ERROR: Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install " + " ".join(missing_deps))
        return
    
    # Create and run the application
    try:
        root = tk.Tk()
        app = BiometricApp(root)
        
        print("Application started successfully")
        print("GUI initialized - you can now register and authenticate users")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {e}")
        print("Please check your installation and try again")


if __name__ == "__main__":
    main()
