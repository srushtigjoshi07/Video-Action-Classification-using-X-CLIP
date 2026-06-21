import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import cv2

# 🔹 LOAD MODEL
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
processor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# 🔹 ADD YOUR VIDEO HERE 👇
video_path = "vdo1.mp4" 

# 🔹 READ VIDEO
cap = cv2.VideoCapture(video_path)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()

# 🔹 CHECK IF VIDEO LOADED
if len(frames) == 0:
    print("❌ Video not loaded. Check path!")
    exit()

# 🔹 PICK 16 FRAMES
num_frames = 16
indices = np.linspace(0, len(frames)-1, num_frames).astype(int)
sampled_frames = [frames[i] for i in indices]

# 🔹 PREPROCESS
inputs = processor(sampled_frames, return_tensors="pt")

# 🔹 RUN MODEL
with torch.no_grad():
    outputs = model(**inputs)

# 🔹 GET RESULT
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
label = model.config.id2label[predicted_class]

print("🎯 Predicted action:", label)