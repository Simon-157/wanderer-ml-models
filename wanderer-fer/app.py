import uuid
import time
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
from threading import Thread
from queue import Queue
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
from analytics.EmotionAnalysis import EmotionTransitionAnalyzer, ExtendedEmotionStorage, FrequencyAnalyzer, TimeBasedAnalyzer
from flask import Flask
from flask_cors import CORS # type: ignore

app = Flask(__name__)
CORS(app, resources={r"/analyze_session": {"origins": "http://localhost:5173"}})

cors_allowed_origins=["http://localhost:5173", "http://172.16.11.84:5173"]
socketio = SocketIO(app, cors_allowed_origins=cors_allowed_origins)

# cors for app
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Load the TensorFlow SavedModel
try:
    model = tf.saved_model.load('saved_model2')
except Exception as e:
    print("Error loading model:", e)
    exit()

# Get the inference function from the SavedModel
infer = model.signatures['serving_default']

# Load Haar cascade classifier for face detection
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Instantiate ExtendedEmotionStorage
storage = ExtendedEmotionStorage()

# Instantiate analyzers
frequency_analyzer = FrequencyAnalyzer(storage)
transition_analyzer = EmotionTransitionAnalyzer(storage)
time_based_analyzer = TimeBasedAnalyzer(storage)

# Queue for processing frames
frame_queue = Queue()

# Global variable for storing error message
error_message = None

def process_frame(frame_data):
    global error_message
    print("Processing frame")
    try:
        # Decode base64 frame data
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert frame to grayscale for face detection
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        # Process each detected face
        for (x, y, w, h) in faces_detected:
            # Crop the face region and resize it to match model input size
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Convert the cropped face region to array and preprocess
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # Perform emotion prediction using the loaded model
            predictions = infer(tf.constant(img_pixels))['dense_1']

            # Get the predicted emotion
            max_index = np.argmax(predictions[0])
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            predicted_emotion = emotions[max_index]

            print("Predicted emotion:", predicted_emotion)
            # Emit the predicted emotion back to the client
            socketio.emit('predicted_emotion', {'emotion': predicted_emotion})

            # Store predicted emotion with timestamp and session ID
            user_id = "user_123"  # TODO: change this
            session_id = "session_123" # Generates a random UUID
            confidence_score = np.max(predictions[0])
            storage.store_emotion(user_id, session_id, predicted_emotion, str(confidence_score))

    except Exception as e:
        error_message = 'Error processing frame: {}'.format(e)

    # Emit the frame processing result back to the client
        socketio.emit('error', {'error_message': error_message})

def frame_processor():
    while True:
        frame_data = frame_queue.get()
        process_frame(frame_data)
        frame_queue.task_done()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(frame_data):
    # Add frame to the queue for processing
    frame_queue.put(frame_data)



def analyze_session(user_id, session_id):
    try:
        # Frequency analysis
        print("Analyzing frequency...")
        emotion_frequency = frequency_analyzer.analyze_frequency(user_id, session_id)
        print("Emotion Frequency:", emotion_frequency)

        # Emotion transition analysis
        print("Analyzing transitions...")
        transitions = transition_analyzer.analyze_transitions(user_id, session_id)
        transitions = {str(key): value for key, value in transitions.items()}  # Convert tuple keys to strings
        print("Emotion Transitions:", transitions)

        # Time-based analysis
        print("Analyzing time-based emotions...")
        time_based_emotions = time_based_analyzer.analyze_time_based(user_id, session_id)
        print("Time-Based Emotions:", time_based_emotions)

        analysis_result = {
            'emotion_frequency': emotion_frequency,
            'transitions': transitions,
            'time_based_emotions': time_based_emotions
        }
        return analysis_result
    except Exception as e:
        print("Error analyzing session:", e)
        emit('error', {'message': 'Error analyzing session'})



@app.route('/analyze_session', methods=['POST'])
def trigger_analysis():
    try:
        user_id = request.json.get('user_id')
        session_id = request.json.get('session_id')
        if user_id is None or session_id is None:
            raise ValueError("Both 'user_id' and 'session_id' must be provided")
        result = analyze_session(user_id, session_id)
        return jsonify(result)
    except Exception as e:
        print("Error triggering analysis:", e)
        return "Error triggering analysis"


# a route to end session

@app.route('/end_session', methods=['POST'])
def end_session():
    try:
        user_id = request.json.get('user_id')
        session_id = request.json.get('session_id')
        if user_id is None or session_id is None:
            raise ValueError("Both 'user_id' and 'session_id' must be provided")
        data = storage.downsample_data(user_id, session_id, 0.5)

        # Save emotions to Firebase
        storage.save_emotions_to_firebase(user_id, session_id, data)

        results = analyze_session(user_id, session_id)
        print('results:', results)
        return jsonify(results)
    except Exception as e:
        print("Error ending session:", e)
        return "Error ending session"
    


@app.route('/get_error')
def get_error():
    global error_message
    if error_message:
        msg = error_message
        error_message = None
        return jsonify({'error': msg})
    else:
        return jsonify({'error': 'No error'})

if __name__ == '__main__':
    # Start frame processing thread
    frame_processor_thread = Thread(target=frame_processor)
    frame_processor_thread.daemon = True
    frame_processor_thread.start()
    
    socketio.run(app, debug=True)

