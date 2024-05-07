# Bird Audio Classification Project

## Overview

The Bird Audio Classification Project aims to develop a machine learning system capable of accurately identifying bird species based on their vocalizations captured in audio recordings. Leveraging deep learning techniques and signal processing algorithms, the system processes audio data, extracts relevant features, and classifies bird species with high accuracy.

## Features

- **Real-time Classification:** The system supports real-time classification of bird species, enabling instant identification of birds in the field.
- **Multi-class Classification:** Capable of classifying multiple bird species simultaneously, providing comprehensive insights into avian biodiversity.
- **Noise Robustness:** The system is robust to background noise and non-bird sounds, ensuring accurate classification in diverse environmental conditions.
- **User-friendly Interface:** A user-friendly interface allows users to easily record bird sounds and receive instant species identification feedback.
- **Scalability:** Designed to scale, allowing for the addition of new bird species and adaptation to different geographic regions and habitats.

## Technologies Used

- **Keras:** Deep learning framework used for model development and training.
- **Librosa:** Python package for audio and music signal processing.
- **Pydub:** Python library for audio file manipulation.
- **Matplotlib:** Plotting library for visualizing spectrogram representations.
- **Seaborn:** Statistical data visualization library for enhanced visualizations.
- **Pandas:** Data manipulation and analysis library for handling datasets.
- **Scikit-learn:** Machine learning library for model evaluation and metrics.
- **Streamlit:** Web application framework for building interactive user interfaces.

## Model Architecture

The model architecture consists of a deep Convolutional Neural Network (CNN) trained on extracted features from audio spectrograms. The CNN processes spectrogram representations of bird calls and classifies them into predefined bird species categories.

## Dataset

The dataset comprises 9000 bird audio recordings, each lasting 3 seconds, collected from various sources. The recordings encompass vocalizations from five distinct species of birds, serving as the foundational data for training and validating the classification system.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/bird-audio-classification.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the project directory:

   ```bash
   cd bird-audio-classification
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

3. Upload an audio file and click the "Predict" button to classify the bird species.

## Future Scope

- Explore fine-grained classification techniques for identifying subspecies or individual variations within species.
- Investigate transfer learning approaches to leverage pre-trained models for bird audio classification tasks.
- Extend the system to analyze bird behavior and vocalizations, providing insights into breeding patterns, foraging behavior, and species interactions.
