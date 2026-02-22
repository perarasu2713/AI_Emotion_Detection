🎭 AI Emotion Detection System

Real-time facial emotion recognition using Deep Learning, OpenCV, and Tkinter GUI








✨ Overview

This project is a real-time AI-based Emotion Detection System that uses a Convolutional Neural Network (CNN) to classify human facial expressions into seven emotions:

😠 Angry

🤢 Disgust

😨 Fear

😄 Happy

😢 Sad

😲 Surprise

😐 Neutral

The application captures live video from your webcam, detects faces, predicts emotions, displays confidence levels, speaks the detected emotion using text-to-speech, and even shows an emotion history graph.

🚀 Features

✅ Real-time face detection using OpenCV
✅ Emotion classification using CNN (TensorFlow/Keras)
✅ Confidence percentage display
✅ Voice feedback (Text-to-Speech)
✅ Emotion history tracking
✅ Graph visualization of emotion distribution
✅ Clean Tkinter GUI interface

🧠 Model Architecture

The model is built using Convolutional Neural Networks (CNN) with the following structure:

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Flatten Layer

Dense (128 neurons)

Dropout (0.5)

Output Layer (7 classes, Softmax)

Loss Function: categorical_crossentropy
Optimizer: Adam

📂 Project Structure
AI-Emotion-Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── model/
│   └── emotion_model.h5
│
├── app.py              # Main GUI Application
├── train_model.py      # Model training script
├── README.md
└── requirements.txt
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/AI-Emotion-Detection.git
cd AI-Emotion-Detection
2️⃣ Install Dependencies
pip install -r requirements.txt

Or install manually:

pip install tensorflow opencv-python numpy pillow matplotlib pyttsx3
🏋️‍♂️ Training the Model

Make sure your dataset is structured like:

dataset/train/Angry
dataset/train/Happy
...
dataset/test/Angry
dataset/test/Happy
...

Then run:

python train_model.py

After training, the model will be saved as:

model/emotion_model.h5
🎥 Running the Application
python app.py

Then:

Click Start Camera 🎬

The system detects faces and predicts emotion

It announces the emotion using voice

Click Show Emotion Graph 📊 to view emotion history

Click Stop Camera 🛑 to close webcam

📊 Emotion Graph

The system stores emotion history during runtime and displays a bar graph showing how frequently each emotion was detected.

This helps analyze emotional trends in real-time.

🛠 Technologies Used

🐍 Python

🔥 TensorFlow / Keras

👁 OpenCV

🖼 Tkinter

📊 Matplotlib

🔊 pyttsx3

📌 Requirements

Python 3.x

Webcam

Minimum 4GB RAM recommended

GPU (optional, for faster training)

🔮 Future Improvements

Improve model accuracy with deeper architecture

Add emotion logging to CSV

Deploy as a web app using Flask or Streamlit

Add real-time dashboard analytics

Support for multiple faces simultaneously

🤝 Contributing

Pull requests are welcome.
If you find bugs or have suggestions, feel free to open an issue.

⭐ Show Some Support

If you like this project, consider giving it a ⭐ on GitHub.

👨‍💻 Author

Developed with passion for AI and Computer Vision.
