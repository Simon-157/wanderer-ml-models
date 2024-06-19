import uuid
import time
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
from analytics.EmotionAnalysis import EmotionTransitionAnalyzer, ExtendedEmotionStorage, FrequencyAnalyzer, TimeBasedAnalyzer


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173", "http://172.16.11.84:5173"])

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

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def process_frame(frame_data):
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
            emit('predicted_emotion', {'emotion': predicted_emotion})

            # Store predicted emotion with timestamp and session ID
            user_id = 'user_123'  # TODO: change this
            session_id = str(uuid.uuid4())  # Generates a random UUID
            confidence_score = np.max(predictions[0])
            storage.store_emotion(user_id, session_id, predicted_emotion, str(confidence_score))

    except Exception as e:
        print("Error processing frame:", e)
        # Handle the error gracefully by emitting an error event to the client
        emit('error', {'message': 'Error processing frame'})

@socketio.on('frame')
def handle_frame(frame_data):
    # Process the frame
    process_frame(frame_data)


def analyze_session(user_id, session_id):
    try:
        # Frequency analysis
        emotion_frequency = frequency_analyzer.analyze_frequency(user_id, session_id)
        print("Emotion Frequency:", emotion_frequency)

        # Emotion transition analysis
        transitions = transition_analyzer.analyze_transitions(user_id, session_id)
        print("Emotion Transitions:", transitions)

        # Time-based analysis
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
        user_id = request.args.get('user_id')
        session_id = request.args.get('session_id')
        result = analyze_session(user_id, session_id)
        
        return jsonify(result)
    except Exception as e:
        print("Error triggering analysis:", e)
        return "Error triggering analysis"

if __name__ == '__main__':
    socketio.run(app, debug=True)
