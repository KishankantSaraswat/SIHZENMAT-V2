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

import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from flask import session, request, jsonify

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for the model and scaler
model_path = os.path.join(BASE_DIR, 'yoga_pose_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'yoga_pose_scaler.pkl')

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

@app.route('/progress-tracking')
def progress_tracking():
    # You can pass dynamic data to the template if required.
    # Example of dummy data (can be replaced with actual data from your database or logic):
    data = {
        "weekly_stats": {
            "practice_time": "4.5 hrs",
            "practice_progress": 75,
            "poses_mastered": "12/15",
            "pose_progress": 80,
            "consistency": 90
        },
        "practice_streak": {
            "days": 7,
            "hours": 4,
            "minutes": 30
        },
        "achievements": [
            {"icon": "✨", "title": "7-Day Streak"},
            {"icon": "✔️", "title": "Perfect Form"}
        ],
        "pose_proficiency": [
            {"pose": "Warrior II", "level": "Advanced"},
            {"pose": "Tree Pose", "level": "Intermediate"}
        ]
    }
    return render_template('progress.html', data=data)
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

# Load environment variables
groq_api_key = os.environ['GROQ_API_KEY']  # Ensure your API key is set in the environment variables

# Initialize Groq Langchain chat object
def get_conversation_chain(model_name):
    memory = ConversationBufferWindowMemory(k=5)  # Default memory length, can be adjusted
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    return ConversationChain(llm=groq_chat, memory=memory)

# Format user input to limit response length and enforce bullet points
def format_user_prompt(question):
    formatted_prompt = f"""
    {question}

    Please provide the answer, and the total response should not exceed 30 words.
    """
    return formatted_prompt

# Process and format the response for user
def process_response(response_text):
    # Check if response_text contains line breaks or bullet points
    points = response_text.split('\n')  # Assuming each point comes on a new line
    
    if len(points) == 1:
        # Try to split by periods if no line breaks are found
        points = response_text.split('. ')
    
    # Clean and reformat the response to remove any unwanted symbols and ensure proper format
    formatted_response = ''
    for i, point in enumerate(points):
        point = point.strip('- ')  # Strip unwanted symbols like dashes or extra spaces
        if point:
            formatted_response += f"{i+1}. {point}<br>"  # Add HTML line break and numbering

    return formatted_response


# Route to display the chatbot UI
@app.route('/chatbot', methods=['GET'])
def chatbot():
    """
    This route serves the chatbot interface when accessed by the user.
    It loads the chatbot page.
    """
    return render_template('home/chatbot.html')

# Route to handle chat requests from the frontend
# Route to handle chat requests from the frontend
@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('message')
    selected_model = "mixtral-8x7b-32768"  # Example model, adjust as needed
    memory_length = 5  # Default memory length

    # Get conversation chain and process the user's message
    conversation_chain = get_conversation_chain(selected_model)
    conversation_chain.memory.k = memory_length

    # Format the question with word limit and bullet points
    formatted_prompt = format_user_prompt(user_question)
    
    try:
        # Invoke the conversation chain
        response = conversation_chain.invoke(formatted_prompt)
        
        # Check if response is valid and process it
        ai_response = process_response(response['response'])

    except Exception as e:
        ai_response = f"Error processing the response: {str(e)}"
        print(f"Error during conversation chain invoke: {str(e)}")

    # Update the chat history stored in the session
    chat_history = session.get('chat_history', [])
    chat_history.append({'human': user_question, 'AI': ai_response})
    session['chat_history'] = chat_history

    return jsonify({'response': ai_response})



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)