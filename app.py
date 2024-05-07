import streamlit as st
import numpy as np
import tensorflow as tf
import seaborn as sns
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import json
from pydub import AudioSegment
import noisereduce as nr

st.set_page_config(layout='wide', page_title="Bird Classification", page_icon="🐦")

def read_file(path):
    y, _ = librosa.load(path)
    return y

def spec_to_db(y):
    y_db = librosa.amplitude_to_db(y, ref=100)
    return y_db

def preprocess_audio(audio_path):
    y, _ = librosa.load(audio_path)
    
    spectrogram = tf.abs(tf.signal.stft(y, frame_length=512, frame_step=64))
    spectrogram_db = spec_to_db(spectrogram)
    spectrogram_db = spectrogram_db / 80 + 1
    spectrogram_db = np.expand_dims(spectrogram_db, axis=0)
    return spectrogram_db

model = tf.keras.models.load_model("savedmodel.h5")

class_names = {
    0: "Bewick's Wren",
    1: "Northern Mockingbird",
    2: "American Robin",
    3: "Song Sparrow",
    4: "Northern Cardinal"
}

def convert_mp3_to_wav(mp3_path, wav_path):
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

def predict_bird_species(uploaded_file, segment_length=3, overlap=1, threshold=0.3):

    if uploaded_file.type == 'audio/mp3':
        mp3_path = "temp.mp3"
        uploaded_file.seek(0)
        with open(mp3_path, 'wb') as f:
            f.write(uploaded_file.read())
        wav_path = "temp.wav"
        convert_mp3_to_wav(mp3_path, wav_path)
        audio_path = wav_path
    else:
        audio_path = uploaded_file
    
    st.audio(audio_path, format='audio/wav', start_time=0)
    
    y, sr = librosa.load(audio_path)
    segment_hop = int(segment_length * sr)
    overlap_hop = int(overlap * sr)
    
    predictions = []
    for i in range(0, len(y) - segment_hop + 1, segment_hop - overlap_hop):
        segment = y[i:i+segment_hop]
        if len(segment) < segment_hop:
            continue
        spectrogram = tf.abs(tf.signal.stft(segment, frame_length=512, frame_step=64))
        spectrogram_db = spec_to_db(spectrogram)
        spectrogram_db = spectrogram_db / 80 + 1
        spectrogram_db = np.expand_dims(spectrogram_db, axis=0)
        prediction = model.predict(spectrogram_db)
        predictions.append(prediction)
    
    if not predictions:
        st.markdown("#### No valid segments found in the audio.")
        return
    
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    
    max_prob = np.max(mean_prediction)
    if max_prob < threshold:
        st.markdown("#### This is not a bird sound")
    else:
        predicted_class_index = np.argmax(mean_prediction)
        predicted_species = class_names[predicted_class_index]
        confidence = max_prob * 100
        st.markdown(f"#### Predicted Bird Species: {predicted_species}")
        st.markdown(f"#### Confidence: {confidence:.2f}%")

    fig, _ = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db[0], sr=sr, x_axis='time', y_axis='linear', cmap="magma")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    st.pyplot(fig)


def display_evaluation():
    st.title("Model Evaluation")
    with open('accuracy.json', 'r') as f:
        accuracy_data = json.load(f)

    st.subheader("Model Evaluation Metrics")

    metrics = {
        "Metric": ["Accuracy", "F1 Score", "Precision", "Recall", "AUC"],
        "Value": [
            accuracy_data['accuracy'],
            accuracy_data['F1'],
            accuracy_data['Precision'],
            accuracy_data['Recall'],
            accuracy_data['AUC']
        ]
    }

    st.table(metrics)

    st.subheader(f"Confusion Matrix:")
    conf_matrix_df = pd.read_csv('confusion_matrix.csv')

    labels = ['Bewick\'s Wren', 'Northern Mockingbird', 'American Robin', 'Song Sparrow', 'Northern Cardinal']

    plt.figure(figsize=(15, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediction')
    plt.ylabel('Label')

    fig, _ = plt.gcf(), plt.gca()
    st.pyplot(fig)


page = st.selectbox("Select a page:", ["Classification", "Evaluation"])
threshold = 0.3
segment_length = 3
overlap = 1

if page == "Classification":
    st.title("Bird Species Classification")
    bird_species = list(class_names.values())
    st.markdown("*Bird species that can be classified: `" + "`, `".join(bird_species) + "`", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    predict_button = st.button("Predict")
    if uploaded_file is not None and predict_button:
        predict_bird_species(uploaded_file, segment_length, overlap, threshold)
else:
    display_evaluation()