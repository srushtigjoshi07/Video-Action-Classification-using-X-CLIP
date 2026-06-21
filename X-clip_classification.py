import torch
import cv2
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel

# ✅ FIX: define model_name first
model_name = "microsoft/xclip-base-patch32"

processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)
model.eval()

labels = [
    "a person walking",
    "a person sitting",
    "a person jumping",
    "a person standing still",
    "a person running",
    "a person waving"
]

video_path = "vdo1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()
print(f"Total frames loaded: {len(frames)}")

MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def build_pixel_values(clip_frames):
    processed = []
    for frame in clip_frames:
        img = frame.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)
    stacked = np.stack(processed, axis=0)
    tensor  = torch.from_numpy(stacked).unsqueeze(0)
    return tensor

text_inputs = processor(text=labels, return_tensors="pt", padding=True)
input_ids      = text_inputs["input_ids"]
attention_mask = text_inputs["attention_mask"]

test_pv = build_pixel_values(frames[:8])
print(f"pixel_values shape: {test_pv.shape}")
assert test_pv.ndim == 5, f"Expected 5D tensor, got {test_pv.ndim}D!"

WINDOW_SIZE = 8
STRIDE      = 4

print("\n🎬 Action Timeline:\n")

for i in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
    clip = frames[i : i + WINDOW_SIZE]
    pixel_values = build_pixel_values(clip)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

    probs      = outputs.logits_per_video.softmax(dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item() * 100

    start_time = i / fps
    end_time   = (i + WINDOW_SIZE) / fps

    print(f"{start_time:5.1f}s – {end_time:5.1f}s │ {labels[pred_idx]:<30} ({confidence:.1f}%)")

print("\n✅ Done.")