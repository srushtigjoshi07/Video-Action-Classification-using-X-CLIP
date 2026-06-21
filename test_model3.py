import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights

# 🔹 Load pretrained model
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

# 🔹 Get labels
labels = weights.meta["categories"]

# 🔹 Video path
video_path = "test(8).mp4"

# 🔹 Read video
cap = cv2.VideoCapture(video_path)
frames = []

while len(frames) < 16:   # R3D needs ~16 frames
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (112, 112))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()

if len(frames) < 16:
    print("❌ Not enough frames")
    exit()

# 🔹 Convert to tensor
frames = np.array(frames)            # (T, H, W, C)
frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
frames = torch.tensor(frames).float() / 255.0

# 🔹 Add batch dimension
frames = frames.unsqueeze(0)  # (1, C, T, H, W)

# 🔹 Predict
with torch.no_grad():
    outputs = model(frames)

# 🔹 Get top prediction
pred = torch.argmax(outputs, dim=1).item()

print("🎯 Predicted action:", labels[pred])