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

# sklearn & joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import joblib

# mediapipe & tts
import mediapipe as mp
import pyttsx3

# TensorFlow for embeddings and improved training
import tensorflow as tf
from tensorflow.keras import layers, models

# ArcFace face recognition
try:
    import insightface
    from insightface.app import FaceAnalysis
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    print("Warning: insightface not available. Install with: pip install insightface onnxruntime-gpu")

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
    
    def add_user(self, name, face_embedding, palm_features=None):
        """Add a new user with face embedding and optional palm features"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Convert numpy arrays to JSON strings
            face_emb_json = json.dumps(face_embedding.tolist())
            palm_features_json = json.dumps(palm_features.tolist()) if palm_features is not None else None
            
            cursor.execute('''
                INSERT INTO users (name, face_embedding, palm_features)
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
                return np.array(json.loads(result[0]))
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
                return np.array(json.loads(result[0]))
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
    """Handles ArcFace face detection and recognition"""
    
    def __init__(self):
        if not ARCFACE_AVAILABLE:
            print("Warning: ArcFace not available. Face recognition will be disabled.")
            self.app = None
            return
        
        try:
            # Initialize ArcFace
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("ArcFace initialized successfully")
        except Exception as e:
            print(f"Failed to initialize ArcFace: {e}")
            self.app = None
    
    def extract_face_embedding(self, image):
        """Extract 512-D face embedding using ArcFace"""
        if self.app is None:
            print("ArcFace not available - using fallback face detection")
            return self._fallback_face_detection(image)
        
        try:
            # Detect faces
            faces = self.app.get(image)
            if len(faces) == 0:
                print("No faces detected by ArcFace - trying fallback")
                return self._fallback_face_detection(image)
            
            # Get the first detected face
            face = faces[0]
            print(f"ArcFace detected face with confidence: {face.det_score:.3f}")
            return face.embedding
        except Exception as e:
            print(f"Face embedding extraction failed: {e}")
            print("Trying fallback face detection...")
            return self._fallback_face_detection(image)
    
    def _fallback_face_detection(self, image):
        """Fallback face detection using OpenCV Haar cascades"""
        try:
            # Convert to grayscale for Haar cascade
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Load pre-trained Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces with multiple scales
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # More sensitive detection
                minNeighbors=3,    # Lower threshold for detection
                minSize=(50, 50),  # Minimum face size
                maxSize=(300, 300) # Maximum face size
            )
            
            if len(faces) == 0:
                print("No faces detected by fallback method")
                return None
            
            # Get the largest face (usually the closest one)
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = int(min(w, h) * 0.1)
            y1 = max(0, y - padding)
            y2 = min(image.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(image.shape[1], x + w + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            # Resize to standard size for embedding
            face_resized = cv2.resize(face_region, (112, 112))
            
            # Create a more stable embedding based on face region characteristics
            # This is a simplified approach - in production, use a proper face embedding model
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Create a pseudo-embedding based on face features
            # This is not as accurate as ArcFace but provides a fallback
            embedding = np.zeros(512, dtype=np.float32)
            
            # Use some basic image statistics as features
            embedding[0] = np.mean(face_normalized)
            embedding[1] = np.std(face_normalized)
            embedding[2] = np.percentile(face_normalized, 25)
            embedding[3] = np.percentile(face_normalized, 75)
            
            # Add some random but consistent features for the rest
            np.random.seed(hash(f"{x}_{y}_{w}_{h}") % 2**32)
            embedding[4:] = np.random.normal(0, 0.1, 508).astype(np.float32)
            
            print(f"Fallback face detection successful - face size: {w}x{h}, position: ({x},{y})")
            return embedding
            
        except Exception as e:
            print(f"Fallback face detection failed: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.6):
        """Compare two face embeddings using cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return False, 0.0
        
        try:
            # Reshape embeddings for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return similarity > threshold, similarity
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Test face detection
            embedding = self.extract_face_embedding(rgb)
            
            if embedding is not None:
                cv2.putText(frame, "FACE DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Embedding shape: {embedding.shape}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw detection box
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, "Position your face in the center", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save test image", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
        print("Face detection test completed")
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
        
        # Camera
        self.cap = None
        
    def register_new_user(self, name, num_face_images=10, num_palm_scans=10):
        """Register a new user with both face and palm biometrics"""
        if not name:
            return False, "Name is required"
        
        # Check if user already exists
        existing_users = self.db_manager.get_all_users()
        if name in existing_users:
            return False, f"User '{name}' already exists"
        
        print(f"Starting registration for user: {name}")
        speak(f"Starting registration for {name}")
        
        # Step 1: Face Registration
        print("Step 1: Face Registration")
        speak("Step 1: Face registration. Look at the camera and move your head slightly.")
        
        face_embeddings = []
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        face_captured = 0
        last_capture_time = 0
        face_detection_count = 0
        
        while face_captured < num_face_images and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try to extract face embedding
            embedding = self.face_system.extract_face_embedding(rgb)
            
            if embedding is not None:
                face_detection_count += 1
                
                # Draw face detection box and instructions
                cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 255, 0), 2)
                cv2.putText(frame, f"Face detected - {face_captured + 1}/{num_face_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Move your head slightly for better coverage", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Auto-capture every 3 seconds if face is detected
                current_time = time.time()
                if current_time - last_capture_time >= 3:  # Capture every 3 seconds
                    face_embeddings.append(embedding)
                    face_captured += 1
                    last_capture_time = current_time
                    speak(f"Face captured {face_captured}")
                    print(f"Face captured {face_captured}/{num_face_images}")
                    
                    # Show capture feedback
                    cv2.imshow("Capture Saved", frame)
                    cv2.waitKey(500)
                    cv2.destroyWindow("Capture Saved")
            else:
                face_detection_count = 0
                cv2.putText(frame, "No face detected - Position your face in the center", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Ensure good lighting and face is clearly visible", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show progress
            cv2.putText(frame, f"Progress: {face_captured}/{num_face_images} | Face detected: {'Yes' if embedding is not None else 'No'}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Face Registration - {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_embeddings) < 5:
            return False, f"Insufficient face captures: {len(face_embeddings)}"
        
        # Calculate average face embedding
        avg_face_embedding = np.mean(face_embeddings, axis=0)
        
        # Step 2: Palm Registration
        print("Step 2: Palm Registration")
        speak("Step 2: Palm registration. Show your right palm.")
        
        palm_features = self.palm_system.scan_palm_for_user(name, num_palm_scans)
        if palm_features == 0:
            return False, "Palm registration failed"
        
        # Get the last captured palm features
        if self.palm_system.features:
            avg_palm_features = np.mean(self.palm_system.features[-num_palm_scans:], axis=0)
        else:
            return False, "No palm features captured"
        
        # Store in database
        success = self.db_manager.add_user(name, avg_face_embedding, avg_palm_features)
        if not success:
            return False, "Failed to save user data"
        
        # Train palm model
        if len(self.palm_system.labels) >= 2:
            self.palm_system.train_model()
        
        speak(f"Registration complete for {name}")
        print(f"User {name} registered successfully")
        return True, f"User {name} registered successfully"
    
    def authenticate_user(self, name):
        """Two-factor authentication: face + palm"""
        if not name:
            return False, "Name is required"
        
        # Check if user exists
        stored_face_embedding = self.db_manager.get_user_face_embedding(name)
        if stored_face_embedding is None:
            return False, f"User '{name}' not found"
        
        print(f"Starting authentication for user: {name}")
        speak(f"Starting authentication for {name}")
        
        # Step 1: Face Authentication
        print("Step 1: Face Authentication")
        speak("Step 1: Face authentication. Look at the camera.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        face_authenticated = False
        face_embedding = None
        start_time = time.time()
        
        while not face_authenticated and (time.time() - start_time) < 30:  # 30 second timeout
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try to extract face embedding
            embedding = self.face_system.extract_face_embedding(rgb)
            
            if embedding is not None:
                # Compare with stored embedding
                is_match, similarity = self.face_system.compare_embeddings(
                    embedding, stored_face_embedding, threshold=0.6
                )
                
                if is_match:
                    face_authenticated = True
                    face_embedding = embedding
                    cv2.putText(frame, f"Face authenticated! Similarity: {similarity:.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, "Authentication successful!", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    speak("Face authentication successful")
                    print(f"Face authenticated with similarity: {similarity:.3f}")
                    
                    # Show success feedback
                    cv2.imshow("Face Authentication Success", frame)
                    cv2.waitKey(1000)
                    cv2.destroyWindow("Face Authentication Success")
                else:
                    cv2.putText(frame, f"Face not recognized. Similarity: {similarity:.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "Try adjusting your position or lighting", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Position your face in the center of the camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show remaining time
            remaining_time = 30 - int(time.time() - start_time)
            cv2.putText(frame, f"Time remaining: {remaining_time}s", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Face Authentication - {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_authenticated:
            return False, "Face authentication failed"
        
        # Step 2: Palm Authentication
        print("Step 2: Palm Authentication")
        speak("Step 2: Palm authentication. Show your right palm.")
        
        palm_authenticated = False
        start_time = time.time()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        while not palm_authenticated and (time.time() - start_time) < 30:  # 30 second timeout
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.palm_system.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand
                if not self.palm_system.is_right_hand(landmarks):
                    cv2.putText(frame, "Please use your RIGHT hand", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(f"Palm Authentication - {name}", frame)
                    cv2.waitKey(1)
                    continue
                
                self.palm_system.mp_drawing.draw_landmarks(frame, landmarks, self.palm_system.hands.HAND_CONNECTIONS)
                
                try:
                    # Extract palm features
                    palm_features = self.palm_system.extract_features(landmarks, frame)
                    
                    # Compare with stored palm features
                    stored_palm_features = self.db_manager.get_user_palm_features(name)
                    if stored_palm_features is not None:
                        # Calculate similarity (Euclidean distance)
                        distance = np.linalg.norm(palm_features - stored_palm_features)
                        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                        
                        if similarity > 0.7:  # Threshold for palm authentication
                            palm_authenticated = True
                            cv2.putText(frame, f"Palm authenticated! Similarity: {similarity:.3f}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            speak("Palm authentication successful")
                            print(f"Palm authenticated with similarity: {similarity:.3f}")
                        else:
                            cv2.putText(frame, f"Palm not recognized. Similarity: {similarity:.3f}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                except Exception as e:
                    print("Palm authentication error:", e)
                
                cv2.putText(frame, "Show your right palm for authentication", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"Palm Authentication - {name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not palm_authenticated:
            return False, "Palm authentication failed"
        
        # Both authentications successful
        speak(f"Authentication successful for {name}")
        print(f"User {name} authenticated successfully")
        return True, f"User {name} authenticated successfully"
    
    def list_users(self):
        """Get list of all registered users"""
        return self.db_manager.get_all_users()
    
    def delete_user(self, name):
        """Delete a user from the system"""
        # Delete from database
        db_success = self.db_manager.delete_user(name)
        
        # Delete from palm system
        palm_success = self.palm_system.delete_user(name)
        
        if db_success and palm_success:
            speak(f"User {name} deleted successfully")
            return True, f"User {name} deleted successfully"
        else:
            return False, f"Failed to completely delete user {name}"


def speak(text):
    """Voice assistant via pyttsx3 (non-blocking)"""
    
    print(text)  # Fallback to print in addition to speech
    def _s():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_s, daemon=True).start()



class PalmBiometricSystem:
    def __init__(self):
        # MediaPipe Hands and Palm
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Palm lines detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Data storage
        self.features = []  # list of np arrays
        self.labels = []    # corresponding labels
        self.embeddings = []  # TensorFlow embeddings
        self.images = []     # Raw images for training
        self.embedding_model = None

        # Model
        self.model = None

        # Flags
        self.capture_running = False
        self.recognize_running = False
        self.auto_training = False

        # Load if available
        self.load_data()
        self.build_embedding_model()

        # Timers for non-repetitive speaking
        self.last_wrong_hand_speak_time = 0

    def build_embedding_model(self):
        """Build a simple CNN model for palm embeddings"""
        input_shape = (128, 128, 3)  # Resized image dimensions
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu')  # Embedding vector
        ])
        
        self.embedding_model = model

    def is_right_hand(self, landmarks):
        """Check if the detected hand is a right hand"""
        # Use wrist and thumb landmarks to determine handedness
        wrist = landmarks.landmark[0]
        thumb_cmc = landmarks.landmark[1]
        pinky_mcp = landmarks.landmark[17]
        
        # For right hand, thumb is on the left side of the hand
        # when palm is facing the camera
        if thumb_cmc.x < wrist.x and pinky_mcp.x > wrist.x:
            return True
        return False

    # -------------------------
    # Feature extraction
    # -------------------------
    def _landmark_to_np(self, landmarks):
        """Convert MediaPipe landmarks into Nx3 numpy array"""
        arr = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr

    def extract_features(self, landmarks, image=None):
        """
        Build a scale-invariant, rotation-robust feature vector:
         - normalized fingertip positions relative to wrist, scaled by palm size
         - pairwise fingertip distances
         - angles between fingers
         - palm width/height ratio
         - palm line features (if image provided)
        Returns a 1D numpy array of fixed length.
        """
        pts = self._landmark_to_np(landmarks)  # shape (21,3)
        wrist = pts[0, :2]

        # Key indices
        tips = [4, 8, 12, 16, 20]    # thumb, index, middle, ring, pinky tips
        bases = [2, 5, 9, 13, 17]    # some base points for palms/fingers

        # Compute palm size as distance between wrist and middle_mcp (9) or bounding box diagonal
        mcp_middle = pts[9, :2]
        palm_size = np.linalg.norm(mcp_middle - wrist) + 1e-6

        # Normalized fingertip positions relative to wrist, scaled by palm_size
        pos_feats = []
        for idx in tips:
            rel = (pts[idx, :2] - wrist) / palm_size
            pos_feats.extend([rel[0], rel[1]])

        # Inter-fingertip pairwise distances (normalized)
        dist_feats = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(pts[tips[i], :2] - pts[tips[j], :2]) / palm_size
                dist_feats.append(d)

        # Angles between vector from wrist to each fingertip (in degrees)
        angle_feats = []
        for idx in tips:
            v = pts[idx, :2] - wrist
            ang = degrees(atan2(v[1], v[0])) / 180.0  # normalize by 180
            angle_feats.append(ang)

        # Palm aspect ratio: width (distance between index_base and pinky_base) / height (wrist to middle_tip)
        index_base = pts[5, :2]
        pinky_base = pts[17, :2]
        middle_tip = pts[12, :2]
        palm_width = np.linalg.norm(index_base - pinky_base)
        palm_height = np.abs(middle_tip[1] - wrist[1]) + 1e-6
        ratio = (palm_width / palm_height)

        # Palm line features (if image provided)
        line_feats = []
        if image is not None:
            try:
                # Extract palm region based on landmarks
                palm_region = self.extract_palm_region(image, landmarks)
                line_feats = self.extract_palm_line_features(palm_region)
            except Exception as e:
                print("Palm line feature extraction failed:", e)
                line_feats = [0] * 10  # Default features if extraction fails

        feats = np.array(pos_feats + dist_feats + angle_feats + [ratio] + line_feats, dtype=np.float32)
        return feats

    def extract_palm_region(self, image, landmarks):
        """Extract the palm region from the image based on landmarks"""
        h, w = image.shape[:2]
        pts = self._landmark_to_np(landmarks)
        
        # Get bounding box around palm (using wrist and finger bases)
        palm_points = pts[[0, 1, 5, 9, 13, 17], :2]  # wrist, thumb cmc, finger mcp joints
        palm_points = (palm_points * [w, h]).astype(np.int32)
        
        # Expand the bounding box a bit
        x_min = max(0, np.min(palm_points[:, 0]) - 20)
        y_min = max(0, np.min(palm_points[:, 1]) - 20)
        x_max = min(w, np.max(palm_points[:, 0]) + 20)
        y_max = min(h, np.max(palm_points[:, 1]) + 20)
        
        # Extract and return the palm region
        return image[y_min:y_max, x_min:x_max]

    def extract_palm_line_features(self, palm_region):
        """Extract features from palm lines using image processing"""
        if palm_region.size == 0:
            return [0] * 10
            
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(palm_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        # Extract line features
        line_count = 0
        avg_length = 0
        avg_angle = 0
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            line_count = len(lines)
            lengths = []
            angles = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                lengths.append(length)
                
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                angles.append(angle)
                
                if -30 <= angle <= 30:
                    horizontal_lines += 1
                elif 60 <= angle <= 120 or -120 <= angle <= -60:
                    vertical_lines += 1
            
            if lengths:
                avg_length = np.mean(lengths)
            if angles:
                avg_angle = np.mean(angles)
        
        # Return a feature vector based on line characteristics
        return [
            line_count, avg_length, avg_angle, 
            horizontal_lines, vertical_lines,
            line_count/100 if line_count > 0 else 0,
            horizontal_lines/line_count if line_count > 0 else 0,
            vertical_lines/line_count if line_count > 0 else 0,
            np.std(lengths) if lengths else 0,
            np.std(angles) if angles else 0
        ]

    def create_embedding(self, image):
        """Create an embedding vector from the palm image using the CNN model"""
        if image is None or self.embedding_model is None:
            return np.zeros(32)  # Return zero vector if no image or model
        
        # Preprocess the image
        resized = cv2.resize(image, (128, 128))
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=0)
        
        # Generate embedding
        embedding = self.embedding_model.predict(expanded, verbose=0)[0]
        return embedding

    # -------------------------
    # Scanning / Registering
    # -------------------------
    def scan_palm_for_user(self, user_id, num_scans=10):
        """
        Open camera, automatically capture images every 2 seconds if hand is detected.
        Stores features, embeddings, and images.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Cannot open camera")
            print("Cannot open camera")
            return 0

        speak(f"Starting capture for {user_id}. Please show your right palm.")
        print(f"Starting capture for {user_id}. Need {num_scans} captures.")
        saved = 0
        start_time = time.time()
        last_capture_time = 0

        while saved < num_scans and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand
                if not self.is_right_hand(landmarks):
                    current_time = time.time()
                    if current_time - self.last_wrong_hand_speak_time >= 5:  # Speak every 5 seconds max
                        speak("Please use your right palm, not left")
                        self.last_wrong_hand_speak_time = current_time
                    cv2.putText(frame, "Please use your RIGHT hand", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(f"Registering: {user_id}", frame)
                    cv2.waitKey(1)
                    continue
                
                self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Auto-capture every 2 seconds if hand is detected
                current_time = time.time()
                if current_time - last_capture_time >= 2:
                    try:
                        feats = self.extract_features(landmarks, frame)
                        embedding = self.create_embedding(frame)
                        
                        self.features.append(feats)
                        self.embeddings.append(embedding)
                        self.images.append(frame)
                        self.labels.append(user_id)
                        
                        saved += 1
                        last_capture_time = current_time
                        
                        speak(f"Captured {saved} for {user_id}")
                        print(f"Captured {saved}/{num_scans}")
                        
                        # Show capture feedback
                        cv2.imshow("Capture Saved", frame)
                        cv2.waitKey(300)
                        cv2.destroyWindow("Capture Saved")
                        
                        # Auto-train if we have enough samples
                        if saved == num_scans and self.auto_training:
                            self.train_model()
                            
                    except Exception as e:
                        print("Feature extraction failed:", e)
                
                cv2.putText(frame, f"Detected - Auto-capturing ({saved}/{num_scans})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"No hand detected - Adjust your palm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(f"Registering: {user_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        duration = time.time() - start_time
        print(f"Finished capturing for {user_id}. Saved: {saved} captures in {duration:.1f}s")
        if saved > 0:
            speak(f"Saved {saved} captures for {user_id}")
        return saved

    # -------------------------
    # Training
    # -------------------------
    def train_model(self, force_grid=False):
        """Train model on stored features and embeddings using TensorFlow."""
        if len(self.labels) < 2:
            speak("Need at least two users to train")
            print("Need at least two different users to train.")
            return False

        # Prepare data
        X_features = np.vstack(self.features)
        X_embeddings = np.vstack(self.embeddings)
        
        # Combine traditional features with embeddings
        X_combined = np.hstack([X_features, X_embeddings])
        y = np.array(self.labels)
        
        # Convert labels to numerical indices
        unique_labels = list(set(self.labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = np.array([label_to_idx[label] for label in self.labels])
        
        # Build a simple neural network classifier
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_combined.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        try:
            history = model.fit(
                X_combined, y_numeric,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            # Store the trained model and label mapping
            self.model = model
            self.label_mapping = unique_labels
            
            # Evaluate the model
            val_acc = history.history['val_accuracy'][-1]
            print(f"Model trained with validation accuracy: {val_acc:.3f}")
            speak(f"Model trained with accuracy {val_acc:.2f}")
            
            return True
        except Exception as e:
            print("Training failed:", e)
            speak("Training failed")
            return False

    # -------------------------
    # Real-time recognition
    # -------------------------
    def recognize_loop(self):
        """Real-time recognition loop using camera. Runs until 'q' pressed or flag turned off."""
        if self.model is None:
            speak("Model not trained yet")
            print("Model not trained. Please train first.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Cannot open camera")
            print("Cannot open camera")
            return

        speak("Starting recognition. Press q to quit.")
        print("Starting recognition. Press 'q' to quit.")
        recent_predictions = []
        
        while self.recognize_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand
                if not self.is_right_hand(lm):
                    current_time = time.time()
                    if current_time - self.last_wrong_hand_speak_time >= 5:  # Speak every 5 seconds max
                        speak("Please use your right palm, not left")
                        self.last_wrong_hand_speak_time = current_time
                    cv2.putText(frame, "Please use your RIGHT hand", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Palm Recognition - press q to quit", frame)
                    cv2.waitKey(1)
                    continue
                
                self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                try:
                    # Extract features and embedding
                    feats = self.extract_features(lm, frame)
                    embedding = self.create_embedding(frame)
                    
                    # Combine features
                    combined = np.hstack([feats, embedding])
                    combined = np.expand_dims(combined, axis=0)
                    
                    # Predict
                    probs = self.model.predict(combined, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    pred = self.label_mapping[pred_idx]
                    conf = probs[pred_idx]
                    
                    recent_predictions.append((pred, conf))
                    # Keep last few predictions to stabilize
                    if len(recent_predictions) > 6:
                        recent_predictions.pop(0)
                    
                    # Voting
                    votes = defaultdict(float)
                    for p, c in recent_predictions:
                        votes[p] += c
                    best = max(votes.items(), key=lambda x: x[1])
                    label, score_sum = best
                    
                    # normalize score
                    score = score_sum / (len(recent_predictions) + 1e-6)
                    cv2.putText(frame, f"{label} ({score:.0%})", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Announce only if high confidence
                    if score > 0.7:
                        speak(f"Recognized {label} with confidence {int(score*100)} percent")
                except Exception as e:
                    print("Recognition feature extraction/prediction error:", e)

            cv2.imshow("Palm Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        speak("Stopping recognition")
        print("Recognition stopped.")

    # -------------------------
    # Save / Load
    # -------------------------
    def save_data(self, features_file=DATA_FILE, model_file=MODEL_FILE, 
                 embeddings_file=EMBEDDINGS_FILE, images_file=IMAGES_FILE, labels_file=LABELS_FILE):
        try:
            # Save features and labels
            with open(features_file, 'wb') as f:
                pickle.dump({'features': self.features, 'labels': self.labels}, f)
            
            # Save embeddings
            np.save(embeddings_file, np.array(self.embeddings))
            
            # Save images (compressed)
            np.savez_compressed(images_file, *self.images)
            
            # Save labels
            np.save(labels_file, np.array(self.labels))
            
            # Save model if it exists
            if self.model is not None:
                self.model.save(model_file)
                
            print(f"Saved data to files")
            speak("Data saved")
            return True
        except Exception as e:
            print("Save failed:", e)
            speak("Save failed")
            return False

    def delete_user(self, user_id):
        """Delete all data for a specific user"""
        if user_id not in self.labels:
            print(f"User {user_id} not found")
            speak(f"User {user_id} not found")
            return False
        
        # Get indices of all samples for this user
        indices = [i for i, label in enumerate(self.labels) if label != user_id]
        
        # Keep only samples that are not from this user
        self.features = [self.features[i] for i in indices]
        self.embeddings = [self.embeddings[i] for i in indices]
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Reset model since data changed
        self.model = None
        
        print(f"Deleted all data for user {user_id}")
        speak(f"Deleted all data for user {user_id}")
        return True

    def load_data(self, features_file=DATA_FILE, model_file=MODEL_FILE, 
                 embeddings_file=EMBEDDINGS_FILE, images_file=IMAGES_FILE, labels_file=LABELS_FILE):
        loaded_any = False
        
        # Load features and labels
        if os.path.exists(features_file):
            try:
                with open(features_file, 'rb') as f:
                    data = pickle.load(f)
                self.features = data.get('features', [])
                self.labels = data.get('labels', [])
                print(f"Loaded features: {len(self.features)} samples, {len(set(self.labels))} users")
                loaded_any = True
            except Exception as e:
                print("Failed to load features:", e)
        else:
            print("Features file not found. Starting fresh.")

        # Load embeddings
        if os.path.exists(embeddings_file):
            try:
                self.embeddings = np.load(embeddings_file, allow_pickle=True).tolist()
                print(f"Loaded embeddings: {len(self.embeddings)}")
                loaded_any = True
            except Exception as e:
                print("Failed to load embeddings:", e)

        # Load images
        if os.path.exists(images_file + '.npz'):
            try:
                loaded_images = np.load(images_file + '.npz', allow_pickle=True)
                self.images = [loaded_images[f'arr_{i}'] for i in range(len(loaded_images.files))]
                print(f"Loaded images: {len(self.images)}")
                loaded_any = True
            except Exception as e:
                print("Failed to load images:", e)

        # Load model
        if os.path.exists(model_file):
            try:
                self.model = tf.keras.models.load_model(model_file)
                print("Loaded trained model.")
                loaded_any = True
                
                # Load label mapping from labels file
                if os.path.exists(labels_file):
                    all_labels = np.load(labels_file, allow_pickle=True)
                    self.label_mapping = list(set(all_labels))
            except Exception as e:
                print("Failed to load model:", e)
        else:
            print("Model file not found; model not loaded.")

        return loaded_any


# -------------------------
# Tkinter GUI wrapper
# -------------------------
class BiometricApp:
    def __init__(self, root):
        self.root = root
        root.title("Two-Factor Biometric Authentication System")
        root.geometry("500x600")
        
        # Initialize the main system
        self.auth_system = BiometricAuthenticationSystem()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Two-Factor Biometric Authentication", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # User Management Section
        user_frame = ttk.LabelFrame(main_frame, text="User Management", padding="10")
        user_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # User selection
        ttk.Label(user_frame, text="Select User:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, state="readonly")
        self.user_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Refresh users button
        ttk.Button(user_frame, text="Refresh Users", command=self.refresh_users).grid(row=0, column=2, padx=(10, 0), pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Register New User", command=self.register_user).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Authenticate User", command=self.authenticate_user).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Delete User", command=self.delete_user).grid(row=0, column=2, padx=5, pady=5)
        
        # Testing and Debug Operations
        test_frame = ttk.LabelFrame(main_frame, text="Testing & Debug", padding="10")
        test_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        test_btn_frame = ttk.Frame(test_frame)
        test_btn_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        ttk.Button(test_btn_frame, text="Test Face Detection", command=self.test_face_detection).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(test_btn_frame, text="Test Dependencies", command=self.test_dependencies).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(test_btn_frame, text="System Info", command=self.show_system_info).grid(row=0, column=2, padx=5, pady=5)
        
        # Palm-only operations (legacy)
        palm_frame = ttk.LabelFrame(main_frame, text="Palm-Only Operations", padding="10")
        palm_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        palm_btn_frame = ttk.Frame(palm_frame)
        palm_btn_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(palm_btn_frame, text="Palm Recognition", command=self.start_palm_recognition).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(palm_btn_frame, text="Save Palm Data", command=self.save_palm_data).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(palm_btn_frame, text="Load Palm Data", command=self.load_palm_data).grid(row=0, column=2, padx=5, pady=5)
        
        # Exit button
        ttk.Button(main_frame, text="Exit", command=self.quit_app, style="Accent.TButton").grid(row=5, column=0, columnspan=2, pady=20)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an operation")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Initialize users list
        self.refresh_users()
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        user_frame.columnconfigure(1, weight=1)
        palm_frame.columnconfigure(1, weight=1)

    def set_status(self, txt):
        self.status_var.set(txt)
        print(txt)
        self.root.update_idletasks()

    def refresh_users(self):
        """Refresh the list of registered users"""
        users = self.auth_system.list_users()
        self.user_combo['values'] = users
        if users:
            self.user_combo.set(users[0])
        self.set_status(f"Found {len(users)} registered users")

    # Two-factor authentication methods
    def register_user(self):
        """Register a new user with face and palm biometrics"""
        name = simpledialog.askstring("User Registration", "Enter user name (unique):", parent=self.root)
        if not name:
            return
        
        self.set_status(f"Starting registration for {name}...")
        self.progress_var.set(0)
        
        def _do():
            try:
                success, message = self.auth_system.register_new_user(name)
                if success:
                    self.set_status(f"Registration successful: {message}")
                    self.refresh_users()
                    messagebox.showinfo("Success", message)
                else:
                    self.set_status(f"Registration failed: {message}")
                    messagebox.showerror("Error", message)
            except Exception as e:
                self.set_status(f"Registration error: {str(e)}")
                messagebox.showerror("Error", f"Registration failed: {str(e)}")
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=_do, daemon=True).start()

    def authenticate_user(self):
        """Authenticate a user using two-factor authentication"""
        selected_user = self.user_var.get()
        if not selected_user:
            messagebox.showwarning("No User Selected", "Please select a user to authenticate.")
            return
        
        self.set_status(f"Starting authentication for {selected_user}...")
        self.progress_var.set(0)
        
        def _do():
            try:
                success, message = self.auth_system.authenticate_user(selected_user)
                if success:
                    self.set_status(f"Authentication successful: {message}")
                    messagebox.showinfo("Success", f"Welcome {selected_user}!")
                else:
                    self.set_status(f"Authentication failed: {message}")
                    messagebox.showerror("Access Denied", message)
            except Exception as e:
                self.set_status(f"Authentication error: {str(e)}")
                messagebox.showerror("Error", f"Authentication failed: {str(e)}")
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=_do, daemon=True).start()

    def delete_user(self):
        """Delete a user from the system"""
        selected_user = self.user_var.get()
        if not selected_user:
            messagebox.showwarning("No User Selected", "Please select a user to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete user '{selected_user}'?"):
            success, message = self.auth_system.delete_user(selected_user)
            if success:
                self.set_status(f"User deleted: {message}")
                self.refresh_users()
                messagebox.showinfo("Success", message)
            else:
                self.set_status(f"Delete failed: {message}")
                messagebox.showerror("Error", message)

    # Palm-only operations (legacy)
    def start_palm_recognition(self):
        """Start palm-only recognition (legacy feature)"""
        if self.auth_system.palm_system.model is None:
            messagebox.showwarning("Model missing", "Please register users first.")
            return
        if self.auth_system.palm_system.recognize_running:
            messagebox.showinfo("Recognition", "Recognition already running.")
            return
        
        self.auth_system.palm_system.recognize_running = True
        self.set_status("Palm recognition running...")
        threading.Thread(target=self.auth_system.palm_system.recognize_loop, daemon=True).start()

    def save_palm_data(self):
        """Save palm data (legacy feature)"""
        ok = self.auth_system.palm_system.save_data()
        if ok:
            self.set_status("Palm data saved")
        else:
            self.set_status("Palm data save failed")

    def load_palm_data(self):
        """Load palm data (legacy feature)"""
        ok = self.auth_system.palm_system.load_data()
        if ok:
            self.set_status("Palm data loaded")
        else:
            self.set_status("No saved palm data found")

    def test_face_detection(self):
        """Test face detection functionality"""
        self.set_status("Testing face detection...")
        threading.Thread(target=self.auth_system.face_system.test_face_detection, daemon=True).start()
    
    def test_dependencies(self):
        """Test all system dependencies"""
        self.set_status("Testing dependencies...")
        threading.Thread(target=self._run_dependency_test, daemon=True).start()
    
    def _run_dependency_test(self):
        """Run dependency test in background"""
        try:
            import subprocess
            import sys
            
            result = subprocess.run([sys.executable, "test_installation.py"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.set_status("Dependency test completed successfully")
                messagebox.showinfo("Dependencies", "All dependencies are working correctly!")
            else:
                self.set_status("Dependency test failed")
                messagebox.showerror("Dependencies", f"Dependency test failed:\n{result.stderr}")
                
        except Exception as e:
            self.set_status(f"Dependency test error: {str(e)}")
            messagebox.showerror("Error", f"Failed to run dependency test: {str(e)}")
    
    def show_system_info(self):
        """Display system information"""
        info = f"""System Information:
        
ArcFace Available: {self.auth_system.face_system.app is not None}
Camera Access: {'Available' if cv2.VideoCapture(0).isOpened() else 'Not Available'}
Database: {self.auth_system.db_manager.db_file}
Registered Users: {len(self.auth_system.list_users())}

Dependencies:
- OpenCV: {'✓' if 'cv2' in sys.modules else '✗'}
- MediaPipe: {'✓' if 'mediapipe' in sys.modules else '✗'}
- TensorFlow: {'✓' if 'tensorflow' in sys.modules else '✗'}
- InsightFace: {'✓' if 'insightface' in sys.modules else '✗'}
- ONNX Runtime: {'✓' if 'onnxruntime' in sys.modules else '✗'}

For face detection issues:
1. Ensure good lighting
2. Position face in center of camera
3. Check camera permissions
4. Try the 'Test Face Detection' button
"""
        
        messagebox.showinfo("System Information", info)

    def quit_app(self):
        """Clean up and exit the application"""
        if hasattr(self.auth_system.palm_system, 'recognize_running'):
            if self.auth_system.palm_system.recognize_running:
                self.auth_system.palm_system.recognize_running = False
                time.sleep(0.5)
        self.root.quit()


def main():
    root = tk.Tk()
    app = BiometricApp(root)
    root.mainloop()


if __name__ == "__main__":

    # If packages missing, the import at top will already fail; handle at runtime.
    try:
        main()
    except Exception as e:
        print("Application error:", e)
        print("Ensure required packages are installed: mediapipe, opencv-python, scikit-learn, joblib, pyttsx3, numpy, tensorflow, insightface, onnxruntime-gpu")
