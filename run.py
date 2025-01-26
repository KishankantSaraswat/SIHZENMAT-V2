import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, render_template, Response
from flask_migrate import Migrate
from flask_minify import Minify
from apps.config import config_dict
from apps import create_app, db

# Paths for the model and scaler
base_path = r'C:\Users\pc\Downloads\flask-soft-dashboard-tailwind-main\flask-soft-dashboard-tailwind-main'
model_path = os.path.join(base_path, 'yoga_pose_model.pkl')
scaler_path = os.path.join(base_path, 'yoga_pose_scaler.pkl')

# Load the saved model and scaler
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# List of yoga poses and their instructions
yoga_poses = {
    "Chair Pose (Utkatasana)": "Stand with your feet together, then bend your knees and lower your hips as if sitting in a chair. Raise your arms overhead.",
    "Tree Pose (Vrikshasana)": "Stand on one leg, place the sole of your other foot on your inner thigh or calf, and bring your hands together at your chest."
}

# Global variables
pose_tracker = deque(maxlen=15)
current_pose = None
smoothed_landmarks = None
CONFIDENCE_THRESHOLD = 0.8

# Function to extract landmarks
def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

# Flask setup
DEBUG = (os.getenv('DEBUG', 'False') == 'True')
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    app_config = config_dict[get_config_mode.capitalize()]
except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

# Route for main interface
@app.route('/')
def index():
    poses = [
        {
            'name': 'Downward Dog', 
            'level': 'Intermediate', 
            'color': 'blue',
            'description': 'A fundamental yoga pose that stretches and strengthens the entire body.'
        },
        {
            'name': 'Warrior Pose', 
            'level': 'Beginner', 
            'color': 'green',
            'description': 'A powerful standing pose that builds strength and improves balance.'
        },
        {
            'name': 'Tree Pose', 
            'level': 'Advanced', 
            'color': 'red',
            'description': 'A balancing pose that improves concentration and core strength.'
        }
    ]
    return render_template('index.html', poses=poses)


@app.route('/test')
def test():
    return render_template('home/test.html')

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    global smoothed_landmarks, current_pose
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract and process landmarks
        landmarks = extract_landmarks(results)
        display_pose = "No Pose Detected"
        confidence = 0.0
        feedback_text = "Position not detected"

        if landmarks is not None:
            # Smoothing the landmarks
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks
            else:
                smoothed_landmarks = 0.3 * landmarks + (1 - 0.3) * smoothed_landmarks

            # Scale the landmarks
            landmarks_scaled = scaler.transform([smoothed_landmarks])

            # Predict the pose
            prediction = model.predict(landmarks_scaled)[0]
            confidence = np.max(model.predict_proba(landmarks_scaled))

            if confidence > CONFIDENCE_THRESHOLD:
                pose_tracker.append(prediction)
                if len(pose_tracker) == pose_tracker.maxlen:
                    current_pose = max(set(pose_tracker), key=pose_tracker.count)
                    display_pose = current_pose

            # Determine feedback based on confidence
            if confidence <= 0.5:
                feedback_text = "Pose not right. Try again."
            elif 0.5 < confidence < 0.8:
                feedback_text = "Good going. Keep trying!"
            else:
                feedback_text = "Excellent form!"

            # Display text on the image
            cv2.putText(image, f"Pose: {display_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, feedback_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)