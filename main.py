import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import pyttsx3
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# Load Model
# -----------------------------
model = load_model("model/emotion_model.h5")

emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]

# Emotion Colors (BGR)
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Neutral': (200, 200, 200)
}

# -----------------------------
# Voice Engine
# -----------------------------
engine = pyttsx3.init()
last_spoken_emotion = ""

# -----------------------------
# Face Detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Tkinter Setup
# -----------------------------
root = tk.Tk()
root.title("AI Emotion Detection - Advanced")
root.geometry("900x750")
root.configure(bg="#1e1e1e")

video_label = Label(root)
video_label.pack(pady=20)

emotion_text = Label(root, text="", font=("Arial", 22),
                     fg="lime", bg="#1e1e1e")
emotion_text.pack()

cap = None
running = False

emotion_history = []

# -----------------------------
# Start Camera
# -----------------------------
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

# -----------------------------
# Stop Camera
# -----------------------------
def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()
    video_label.config(image="")

# -----------------------------
# Show Emotion Graph
# -----------------------------
def show_graph():
    if not emotion_history:
        return

    counts = Counter(emotion_history)

    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.title("Emotion History")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# Update Frame
# -----------------------------
def update_frame():
    global cap, running, last_spoken_emotion

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        prediction = model.predict(roi_gray, verbose=0)
        max_index = np.argmax(prediction)
        emotion = emotion_labels[max_index]
        confidence = int(np.max(prediction) * 100)

        label = f"{emotion} ({confidence}%)"
        emotion_text.config(text=label)

        # Store history
        emotion_history.append(emotion)

        # Speak only if emotion changes
        if emotion != last_spoken_emotion:
            engine.say(f"You look {emotion}")
            engine.runAndWait()
            last_spoken_emotion = emotion

        color = emotion_colors.get(emotion, (0,255,0))

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((700, 500))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(10, update_frame)

# -----------------------------
# Buttons
# -----------------------------
Button(root, text="Start Camera",
       command=start_camera,
       font=("Arial", 14),
       bg="green", fg="white",
       width=15).pack(pady=5)

Button(root, text="Stop Camera",
       command=stop_camera,
       font=("Arial", 14),
       bg="red", fg="white",
       width=15).pack(pady=5)

Button(root, text="Show Emotion Graph",
       command=show_graph,
       font=("Arial", 14),
       bg="blue", fg="white",
       width=20).pack(pady=10)

# -----------------------------
# Run App
# -----------------------------
root.mainloop()
