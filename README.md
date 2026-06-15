# Deepfake-Audio-Detection

## Project Description

This project detects whether an audio file is **Genuine (Human Speech)** or **Deepfake (AI-Generated Speech)**. The system converts audio into Mel Spectrograms and uses a Convolutional Neural Network (CNN) for classification.

## Methodology

1. Audio preprocessing (16 kHz, mono, 3-second clips)
2. Mel Spectrogram extraction
3. CNN-based feature learning
4. Binary classification (Genuine/Fake)

## Pipeline

```text
Audio File (.wav)
        ↓
Preprocessing
        ↓
Mel Spectrogram
        ↓
CNN Model
        ↓
Prediction
(Genuine / Fake)
```

## Model Architecture

* 4 Convolutional Layers
* ReLU Activation
* Max Pooling
* Adaptive Average Pooling
* Fully Connected Layers
* Dropout (0.4)

## Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **F1 Score**
* **Equal Error Rate (EER)**
* **Per-Class Accuracy**

## Running the Project

### Train the Model

```bash
python train_pipeline.py
```

### Evaluate the Model

```bash
python predict.py
```

### Launch Web App

```bash
streamlit run app.py
```

## Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Scikit-learn
* Streamlit

## Outcome

The proposed CNN-based system effectively distinguishes between genuine and AI-generated speech, providing a lightweight and deployable solution for deepfake audio detection.

Kunal Mahawar
ECE,25116059
IIT Roorkee
