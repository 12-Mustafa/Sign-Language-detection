# Sign Language Detection

A Flask web app that uses computer vision to help people communicate. It recognises hand gestures and the sign language alphabet, speaks gestures out loud, transcribes speech, and detects facial emotions.

## Features

| Feature | What it does | Model |
|---|---|---|
| **Gesture to voice** | Reads hand gestures from your webcam and speaks a phrase for each one | MediaPipe hand landmarks + Keras (`mp_hand_gesture/`) |
| **Alphabet recognition** | Predicts the letter (A–Z) shown in an uploaded hand-sign image | CNN built with TFLearn, trained on about 5,200 images |
| **Voice to text** | Listens through your microphone and prints what you say | Google Speech Recognition |
| **Emotion detection** | Detects your face on the webcam and labels it with one of 7 emotions | Haar cascade + Keras CNN (`model/`) |

Each gesture speaks a phrase:

| Gesture | Spoken |
|---|---|
| zero | "stop" |
| one / two / three / four | the number |
| super | "I am fine" |
| up / down | "go up" / "go down" |
| call | "call me" |
| smile | "smile please" |

## Project structure

```
sign_language_recognition/
├── app.py                  # Flask app: login, signup, feature routes
├── gesture_to_voice.py     # Webcam gesture recognition + text-to-speech
├── voice_to_text.py        # Microphone speech-to-text
├── TestEmotionDetector.py  # Webcam emotion detection
├── TrainEmotionDetector.py # Trains the emotion model (expects data/train, data/test)
├── gesture.names           # Gesture class labels
├── mp_hand_gesture/        # Trained gesture model
├── model/                  # Trained emotion model (.json + .h5)
├── haarcascades/           # Face detector
├── indian/                 # Alphabet dataset (train/test), model and train_model.py
├── templates/              # HTML pages
└── static/                 # CSS, JS, images
```

## Setup

You need Python 3.8 (TFLearn doesn't work with recent TensorFlow), a webcam and a microphone.

```bash
git clone https://github.com/12-Mustafa/Sign-Language-detection.git
cd Sign-Language-detection/sign_language_recognition

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install flask opencv-python numpy tensorflow keras tflearn mediapipe \
            pillow tqdm matplotlib gtts pygame mutagen \
            SpeechRecognition pyttsx3 pyaudio
```

## Usage

```bash
python app.py
```

1. Open http://127.0.0.1:5000/signup and create an account. This also creates `user_data.db`.
2. Log in at http://127.0.0.1:5000.
3. Choose a feature:
   - **Gesture Detection**: opens a webcam window. Press `q` to close it.
   - **Text to Speech**: starts voice-to-text in the terminal.
   - **Emotion Detection**: opens a webcam window. Press `q` to close it.
   - **Image Preprocessing**: enter the name of an image from `indian/test/` (e.g. `a (1).jpg`) to predict its letter.

You can also run any script on its own, e.g. `python gesture_to_voice.py`.

### Retrain the alphabet model

```bash
cd indian
python train_model.py
```

## Known issues

- Login and signup use string-built SQL queries and store passwords in plain text. Fine for a local demo, not safe to deploy.
- `TestEmotionDetector.py` refers to an undefined variable (`predicted`) in its stress check, so it crashes after the first frame.
- `app.py` looks for the alphabet model in its own folder, but the model files are in `indian/` (and the `.meta` file is named `... - Copy.meta`). Copy or rename them so the trained weights load.
