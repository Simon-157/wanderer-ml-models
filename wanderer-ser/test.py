import pyaudio
import wave
import librosa
import numpy as np
from keras.models import model_from_json
from keras_self_attention import SeqSelfAttention
from scipy.io.wavfile import write
from scipy.signal import resample
from tensorflow.keras.utils import to_categorical

# Load the saved model
json_file = open('model_json2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'SeqSelfAttention': SeqSelfAttention})
loaded_model.load_weights("saved_models2/best_model.keras")  # Modify path if needed

# Define emotion labels
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def record_audio(filename, duration=10, fs=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = fs
    RECORD_SECONDS = duration
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    print("* Recording audio...")
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* Done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def predict_emotion(audio_file, n_mfcc=30):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Resample if needed
    if sr != 44100:
        y = resample(y, int(len(y) * 44100 / sr))
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=n_mfcc)
    
    # Pad or truncate to match the expected number of frames
    expected_frames = 216
    if mfccs.shape[1] < expected_frames:
        mfccs = np.pad(mfccs, ((0, 0), (0, expected_frames - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > expected_frames:
        mfccs = mfccs[:, :expected_frames]
    
    # Normalize MFCC features
    mfccs_norm = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # batch dimension and channel dimension
    mfccs_norm = np.expand_dims(mfccs_norm, axis=0)
    mfccs_norm = np.expand_dims(mfccs_norm, axis=-1)
    
    # Predict emotion
    prediction = loaded_model.predict(mfccs_norm)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    return emotion

if __name__ == "__main__":
    filename = "recorded_audio.wav"
    record_audio(filename)
    emotion = predict_emotion(filename)
    print("Predicted emotion:", emotion)
