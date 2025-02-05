import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response

app = Flask(name)

# Load Model and Scaler
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Initialize Mediapipe Pose Detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load Yoga Pose Models
models = {
    "tree": load_model_and_scaler("C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_pose_tree_model.pkl", 
                                   "C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_scaler_tree.pkl"),
    "chair": load_model_and_scaler("C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_pose_chair_model.pkl", 
                                    "C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_scaler_chair.pkl"),
    "cobra": load_model_and_scaler("C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_pose_cobra_model.pkl", 
                                    "C:\\Users\\pc\\Downloads\\flask-soft-dashboard-tailwind-main\\flask-soft-dashboard-tailwind-main\\yoga_scaler_cobra.pkl")
}

# Open Webcam
cap = cv2.VideoCapture(0)

# Function to Process Pose
def process_pose(frame, model, scaler):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        data = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
        
        if len(data) == 99:
            data = np.array(data).reshape(1, -1)
            data = scaler.transform(data)
            probabilities = model.predict_proba(data)[0]
            max_prob_index = np.argmax(probabilities)
            max_prob = probabilities[max_prob_index]
            pose_label = model.classes_[max_prob_index]

            if max_prob > 0.85:
                cv2.putText(frame, f"{pose_label} ({max_prob*100:.2f}%)", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Uncertain", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame

# Function to Stream Video
def generate_frames(model, scaler):
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_pose(frame, model, scaler)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect/<pose_name>")
def detect_pose(pose_name):
    if pose_name not in models:
        return "Invalid Pose", 404
    return render_template("pose.html", pose_name=pose_name)

@app.route("/video_feed/<pose_name>")
def video_feed(pose_name):
    if pose_name not in models:
        return "Invalid Pose", 404
    return Response(generate_frames(*models[pose_name]), mimetype='multipart/x-mixed-replace; boundary=frame')

if name == "main":
    app.run(debug=True)  