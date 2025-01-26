from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

app = Flask(__name__, template_folder="apps/templates")

# Load the saved model and scaler
with open(r'C:\Users\pc\Downloads\flask-soft-dashboard-tailwind-main\flask-soft-dashboard-tailwind-main\yoga_pose_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open(r'C:\Users\pc\Downloads\flask-soft-dashboard-tailwind-main\flask-soft-dashboard-tailwind-main\yoga_pose_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Video streaming generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame (similar to your existing code)
        # Convert the frame to RGB and process it
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )
        
        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # Create an HTML template to render the page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
