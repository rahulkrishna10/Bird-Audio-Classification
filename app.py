from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("saved_model.h5")

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

# Define class_names (replace with your actual class names)
class_names = ["Bewick's Wren", "Northern Mockingbird", "American Robin", "Song Sparrow", "Northern Cardinal"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', class_name='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', class_name='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = "uploads/" + filename
        file.save(file_path)

        mel_spec = process_audio(file_path)
        mel_spec = np.expand_dims(mel_spec, axis=0)

        prediction = model.predict(mel_spec)
        predicted_class = np.argmax(prediction)
        class_name = class_names[predicted_class]

        return render_template('result.html', class_name=class_name)

    return render_template('result.html', class_name='Invalid file')

def process_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=10)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr) 
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

if __name__ == '__main__':
    app.run(debug=True)
