from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import tempfile
import os

app = Flask(__name__)

# Ruta para servir la página principal
@app.route('/')
def index():
    return render_template('index.html')

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def calculate_bpm(signal, fps):
    filtered_signal = apply_bandpass_filter(signal, 0.8, 3.0, fps)
    peaks, _ = find_peaks(filtered_signal, distance=fps/2)
    bpm = len(peaks) * (60.0 / (len(signal) / fps))
    return bpm

@app.route('/process-video', methods=['POST'])
def process_video():
    file = request.files['video']
    if file:
        # Save the uploaded video file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            file.save(temp_file.name)
            video_path = temp_file.name
        
        # Process video to calculate BPM
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        green_channel_means = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = frame[:, :, 1]  # Extract green channel
            mean_intensity = np.mean(roi)
            green_channel_means.append(mean_intensity)
        
        signal = np.array(green_channel_means)
        bpm = calculate_bpm(signal, fps)
        cap.release()

        # Clean up temporary file
        os.remove(video_path)

        return jsonify({'bpm': bpm})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
