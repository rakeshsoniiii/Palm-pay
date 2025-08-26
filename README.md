Chat bot https://github.com/Apoorv479/AI-Agent-chat-bot/tree/main

libraries to install for docker.py
pip install pillow opencv-python mediapipe numpy opencv-contrib-python

# Biometric Authentication System

A comprehensive two-factor authentication system using face recognition (ArcFace) and palm recognition.

## Features

- **Face Recognition**: Uses InsightFace with ArcFace model for 512-D embedding extraction
- **Palm Recognition**: Uses MediaPipe for hand tracking and custom feature extraction
- **Two-Factor Authentication**: Combines face and palm verification for enhanced security
- **SQLite Database**: Stores user face embeddings securely
- **Voice Feedback**: Provides audio guidance throughout the process
- **User-Friendly GUI**: Easy-to-use interface with separate tabs for registration, login, and management

## Installation Requirements

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Webcam

### Library Installation

#### Method 1: Using requirements.txt

1. Create a `requirements.txt` file with the following content:

```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
joblib>=1.0.0
pyttsx3>=2.90
tensorflow>=2.5.0
insightface>=0.2.1
mediapipe>=0.8.0
scipy>=1.7.0
matplotlib>=3.3.0
Pillow>=8.0.0
pyautogui>=0.9.0
pandas>=1.3.0
```

2. Install all packages:
```bash
pip install -r requirements.txt
```

#### Method 2: Manual Installation

Install each library individually:

```bash
# Core computer vision and machine learning
pip install opencv-python opencv-contrib-python
pip install numpy scipy scikit-learn joblib

# Deep learning and face recognition
pip install tensorflow insightface

# Hand tracking and palm recognition
pip install mediapipe

# Voice feedback and GUI
pip install pyttsx3 pyautogui

# Utilities
pip install matplotlib Pillow pandas
```

### Platform-Specific Instructions

#### Windows
```bash
# Additional dependency for pyttsx3 on Windows
pip install pypiwin32
```

#### macOS
```bash
# Install system dependencies for pyttsx3
brew install portaudio
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
sudo apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev espeak

# Install Python packages
pip install -r requirements.txt
```

### Verification

After installation, verify all packages are working correctly:

1. Create a test script `test_imports.py`:
```python
import cv2
import numpy as np
from sklearn import svm
import joblib
import pyttsx3
import tensorflow as tf
import insightface
import mediapipe as mp
import sqlite3

print("All imports successful!")
```

2. Run the test:
```bash
python test_imports.py
```

## Library Descriptions

### Core Libraries

1. **opencv-python** (4.5.0+)
   - Computer vision library for image processing
   - Used for camera capture, image manipulation, and display

2. **numpy** (1.19.0+)
   - Numerical computing library
   - Essential for array operations and mathematical computations

3. **scikit-learn** (0.24.0+)
   - Machine learning library
   - Used for SVM classifier and other ML algorithms

4. **joblib** (1.0.0+)
   - Utilities for pipelining Python jobs
   - Used for model serialization and persistence

### Deep Learning & Face Recognition

5. **tensorflow** (2.5.0+)
   - Deep learning framework
   - Used for building and training neural networks

6. **insightface** (0.2.1+)
   - State-of-the-art face recognition library
   - Provides ArcFace model for 512-D face embeddings

### Hand Tracking & Palm Recognition

7. **mediapipe** (0.8.0+)
   - Cross-platform framework for building multimodal applied ML pipelines
   - Used for hand tracking and landmark detection

### Voice & GUI

8. **pyttsx3** (2.90+)
   - Text-to-speech conversion library
   - Provides voice feedback throughout the authentication process

9. **pyautogui** (0.9.0+)
   - Cross-platform GUI automation
   - Used for additional GUI functionalities

### Utilities

10. **scipy** (1.7.0+)
    - Scientific computing library
    - Used for advanced mathematical functions

11. **matplotlib** (3.3.0+)
    - Plotting library
    - Used for visualization (if needed)

12. **Pillow** (8.0.0+)
    - Python Imaging Library
    - Used for image processing tasks

13. **pandas** (1.3.0+)
    - Data analysis library
    - Used for data manipulation and analysis

## Troubleshooting

### Common Issues

1. **OpenCV installation issues**:
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python
   ```

2. **TensorFlow compatibility issues**:
   ```bash
   pip install tensorflow==2.8.0
   ```

3. **MediaPipe installation issues**:
   ```bash
   pip install --upgrade pip
   pip install mediapipe
   ```

4. **InsightFace model download**:
   - The first time you run the application, it will automatically download the ArcFace model
   - Ensure you have internet connection for the initial setup

5. **Webcam access issues**:
   - Ensure no other application is using the webcam
   - Grant camera permissions to your Python environment

### Performance Optimization

For better performance, especially on CPU-only systems:

1. Reduce the input resolution in the code
2. Use fewer samples during registration
3. Adjust the confidence thresholds

## Usage

After installing all dependencies, run the application:

```bash
python biometric_system.py
```

## File Structure

```
biometric_authentication/
├── biometric_system.py      # Main application file
├── requirements.txt         # Dependencies list
├── test_imports.py         # Import verification script
├── biometric_db.sqlite     # SQLite database (created automatically)
├── palm_features.pkl       # Palm features (created automatically)
├── palm_model.joblib       # Palm model (created automatically)
└── README.md               # This file
```

## Support

If you encounter any issues:

1. Check that all dependencies are installed correctly
2. Verify your webcam is working and accessible
3. Ensure you have sufficient lighting for face and palm recognition
4. Check the console for error messages




