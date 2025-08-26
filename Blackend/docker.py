import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import sqlite3
import numpy as np
import json
import os

# --- Constants ---
DB_FILE = "palm_biometrics.db"
SIMILARITY_THRESHOLD = 0.98  # Threshold for cosine similarity matching

# --- Database Management ---
class DatabaseManager:
    """Handles all database operations for the biometric system."""
    def __init__(self, db_file):
        """Initializes the database connection and creates the users table if it doesn't exist."""
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.create_table()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def create_table(self):
        """Creates the 'users' table to store user details and palm features."""
        query = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            dob TEXT NOT NULL,
            aadhaar TEXT UNIQUE NOT NULL,
            pan TEXT UNIQUE NOT NULL,
            palm_features TEXT NOT NULL
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")

    def insert_user(self, full_name, dob, aadhaar, pan, palm_features):
        """Inserts a new user record into the database."""
        # Serialize the numpy array of features into a JSON string
        features_json = json.dumps(palm_features.tolist())
        query = """
        INSERT INTO users (full_name, dob, aadhaar, pan, palm_features)
        VALUES (?, ?, ?, ?, ?);
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (full_name, dob, aadhaar, pan, features_json))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
             # This error occurs if Aadhaar or PAN is not unique
            return -1
        except sqlite3.Error as e:
            print(f"Error inserting user: {e}")
            return None

    def get_all_users_features(self):
        """Retrieves the ID and palm features for all users in the database."""
        query = "SELECT id, palm_features FROM users;"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            # Deserialize JSON string features back into numpy arrays
            return [(row[0], np.array(json.loads(row[1]))) for row in rows]
        except sqlite3.Error as e:
            print(f"Error fetching all users: {e}")
            return []

    def get_user_details_by_id(self, user_id):
        """Retrieves the full details of a user by their ID."""
        query = "SELECT full_name, dob, aadhaar, pan FROM users WHERE id = ?;"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error fetching user details: {e}")
            return None

    def __del__(self):
        """Closes the database connection upon object destruction."""
        if self.conn:
            self.conn.close()

# --- Palm Feature Extraction ---
class PalmFeatureExtractor:
    """Uses Mediapipe to detect hand landmarks and extract a normalized feature vector."""
    def __init__(self):
        """Initializes the Mediapipe Hands model."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_features(self, frame_rgb):
        """
        Processes an RGB frame to find hand landmarks and compute a feature vector.
        The feature vector is normalized to be invariant to hand size and position.
        """
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None, None # No hand detected

        # Assuming only one hand is detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # --- Normalization ---
        # Use the wrist (landmark 0) as the origin.
        origin = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
        
        # Use the distance between wrist (0) and middle finger MCP (9) as the scale.
        # This makes the features robust to the hand's distance from the camera.
        p9 = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
        scale = np.linalg.norm(p9 - origin)
        if scale < 1e-6: # Avoid division by zero
            return None, None

        # Calculate normalized coordinates for all 21 landmarks
        feature_vector = []
        for landmark in hand_landmarks.landmark:
            normalized_point = (np.array([landmark.x, landmark.y]) - origin) / scale
            feature_vector.extend(normalized_point)
            
        return np.array(feature_vector), hand_landmarks

    def draw_landmarks(self, frame, landmarks):
        """Draws the detected landmarks and connections on the frame."""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_hands.HAND_CONNECTIONS
            )
        return frame

# --- Main Application GUI ---
class PalmBiometricApp:
    """The main Tkinter application class."""
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700")

        # --- Initialize components ---
        self.db_manager = DatabaseManager(DB_FILE)
        self.feature_extractor = PalmFeatureExtractor()
        self.cap = cv2.VideoCapture(0)

        # --- State variables ---
        self.current_mode = "idle"  # Modes: "idle", "register", "recognize"
        self.captured_features = None

        # --- GUI Setup ---
        self.setup_gui()

        # Start the video update loop
        self.update()

    def setup_gui(self):
        """Creates and places all the GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Left side: Camera feed and status
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

        # Video feed label
        self.video_label = ttk.Label(left_frame, text="Initializing Camera...")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Status label
        self.status_label = ttk.Label(left_frame, text="Welcome! Please select an option.", anchor="center", font=("Helvetica", 12))
        self.status_label.pack(fill=tk.X, pady=10)

        # Right side: Controls and information display
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False) # Prevent frame from shrinking

        # Control buttons
        ttk.Button(right_frame, text="Register New User", command=self.start_registration).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="Recognize User", command=self.start_recognition).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="Exit", command=self.on_closing).pack(fill=tk.X, pady=5)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=20)

        # Information display area
        self.info_frame = ttk.LabelFrame(right_frame, text="User Information")
        self.info_frame.pack(expand=True, fill=tk.BOTH)
        
        self.info_text = tk.Label(self.info_frame, text="Details will be shown here.", justify=tk.LEFT, anchor="nw", wraplength=230, font=("Helvetica", 11))
        self.info_text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)


    def start_registration(self):
        """Initiates the registration process."""
        self.current_mode = "register"
        self.status_label.config(text="Mode: Register. Please show your left palm.")
        self.clear_info_display()

    def start_recognition(self):
        """Initiates the recognition process."""
        self.current_mode = "recognize"
        self.status_label.config(text="Mode: Recognize. Please show your palm.")
        self.clear_info_display()

    def update(self):
        """Main loop to update the camera feed and handle logic."""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Error: Cannot read from camera.")
            return

        # Flip for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features, landmarks = self.feature_extractor.extract_features(frame_rgb)

        # Draw landmarks on the original BGR frame
        frame = self.feature_extractor.draw_landmarks(frame, landmarks)
        
        # --- Handle logic based on the current mode ---
        if features is not None:
            if self.current_mode == "register":
                self.captured_features = features
                self.current_mode = "idle" # Stop further captures
                self.status_label.config(text="Palm captured successfully! Opening registration form...")
                self.window.after(500, self.open_registration_form)

            elif self.current_mode == "recognize":
                self.perform_recognition(features)
                self.current_mode = "idle" # Stop further recognitions

        # Update the video feed in the GUI
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.config(image=self.photo)

        # Repeat the update loop
        self.window.after(10, self.update)

    def perform_recognition(self, current_features):
        """Compares captured features against the database."""
        self.status_label.config(text="Comparing palm features...")
        all_users = self.db_manager.get_all_users_features()

        if not all_users:
            self.status_label.config(text="No users in the database. Please register first.")
            return

        best_match_id = -1
        max_similarity = -1

        # Cosine Similarity Calculation
        for user_id, stored_features in all_users:
            similarity = np.dot(current_features, stored_features) / (np.linalg.norm(current_features) * np.linalg.norm(stored_features))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = user_id

        if max_similarity > SIMILARITY_THRESHOLD:
            user_details = self.db_manager.get_user_details_by_id(best_match_id)
            if user_details:
                name, dob, aadhaar, pan = user_details
                info = f"Match Found!\n\nName: {name}\nDOB: {dob}\nAadhaar: {aadhaar}\nPAN: {pan}"
                self.update_info_display(info)
                self.status_label.config(text=f"Recognition successful! (Similarity: {max_similarity:.2f})")
        else:
            self.status_label.config(text=f"No match found. (Highest similarity: {max_similarity:.2f})")
            self.clear_info_display()

    def open_registration_form(self):
        """Opens a Toplevel window to get user details."""
        self.reg_window = tk.Toplevel(self.window)
        self.reg_window.title("New User Registration")

        form_frame = ttk.Frame(self.reg_window, padding="20")
        form_frame.pack()
        
        # Form fields
        ttk.Label(form_frame, text="Full Name:").grid(row=0, column=0, sticky="w", pady=5)
        self.name_entry = ttk.Entry(form_frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Date of Birth (YYYY-MM-DD):").grid(row=1, column=0, sticky="w", pady=5)
        self.dob_entry = ttk.Entry(form_frame, width=30)
        self.dob_entry.grid(row=1, column=1, pady=5)

        ttk.Label(form_frame, text="Aadhaar Card Number:").grid(row=2, column=0, sticky="w", pady=5)
        self.aadhaar_entry = ttk.Entry(form_frame, width=30)
        self.aadhaar_entry.grid(row=2, column=1, pady=5)
        
        ttk.Label(form_frame, text="PAN Card Number:").grid(row=3, column=0, sticky="w", pady=5)
        self.pan_entry = ttk.Entry(form_frame, width=30)
        self.pan_entry.grid(row=3, column=1, pady=5)

        # Submit button
        submit_button = ttk.Button(form_frame, text="Submit Registration", command=self.save_registration)
        submit_button.grid(row=4, columnspan=2, pady=20)

        self.reg_window.transient(self.window)
        self.reg_window.grab_set()
        self.window.wait_window(self.reg_window)


    def save_registration(self):
        """Validates and saves the user data from the registration form."""
        name = self.name_entry.get()
        dob = self.dob_entry.get()
        aadhaar = self.aadhaar_entry.get()
        pan = self.pan_entry.get()

        if not all([name, dob, aadhaar, pan, self.captured_features is not None]):
            messagebox.showerror("Error", "All fields are required!", parent=self.reg_window)
            return

        user_id = self.db_manager.insert_user(name, dob, aadhaar, pan, self.captured_features)

        if user_id == -1:
            messagebox.showerror("Error", "A user with this Aadhaar or PAN number already exists.", parent=self.reg_window)
        elif user_id:
            messagebox.showinfo("Success", f"User '{name}' registered successfully with ID: {user_id}", parent=self.reg_window)
            self.reg_window.destroy()
            self.status_label.config(text="Registration complete. Ready for next operation.")
        else:
            messagebox.showerror("Error", "Failed to register user. Check console for details.", parent=self.reg_window)
        
        self.captured_features = None # Clear features after attempting to save

    def update_info_display(self, text):
        """Updates the user information display panel."""
        self.info_text.config(text=text)

    def clear_info_display(self):
        """Clears the user information display panel."""
        self.info_text.config(text="Details will be shown here.")

    def on_closing(self):
        """Handles the application closing event."""
        self.cap.release()
        self.window.destroy()


# --- Main execution block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PalmBiometricApp(root, "Palm Biometric Recognition System")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
