# 🎬 Video Action Classification using X-CLIP

A zero-shot video classification system that identifies human actions directly from video clips using Microsoft's X-CLIP transformer model.

This project processes video frames, extracts temporal information from short clips, and classifies actions without task-specific training by leveraging natural language descriptions of actions.

---

## 🚀 Features

- 🎥 Video-based action recognition
- 🧠 Zero-shot classification using X-CLIP
- ⏱️ Sliding-window temporal analysis
- 📊 Confidence score prediction
- 🔄 Supports multiple action classes
- ⚡ No custom model training required

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Microsoft X-CLIP
- OpenCV
- NumPy

---

## ⚙️ Workflow

```text
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Frame Preprocessing
      │
      ▼
Sliding Window Clip Generation
      │
      ▼
X-CLIP Transformer
      │
      ▼
Action Classification
      │
      ▼
Timeline Predictions
```

---

## 🧠 Model Used

### Microsoft X-CLIP

```python
microsoft/xclip-base-patch32
```

X-CLIP is a transformer-based video understanding model that extends CLIP for video classification by learning both spatial and temporal representations.

Unlike conventional video classifiers, X-CLIP can perform zero-shot action recognition using natural language prompts.

---

## 📋 Action Labels

The project currently classifies the following actions:

```text
a person walking
a person sitting
a person jumping
a person standing still
a person running
a person waving
```

Additional action classes can be added easily by modifying the label list.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/video-action-classification-xclip.git
cd video-action-classification-xclip
```

Install dependencies:

```bash
pip install torch transformers opencv-python numpy
```

---

## ▶️ Usage

Place your video file in the project directory.

```text
vdo1.mp4
```

Run:

```bash
python X-clip_classification.py
```

---

## 📊 Sample Output

```text
🎬 Action Timeline:

0.0s – 2.0s │ a person walking        (94.6%)
1.0s – 3.0s │ a person running        (91.2%)
2.0s – 4.0s │ a person waving         (89.5%)

✅ Done.
```

The model outputs:

- Start timestamp
- End timestamp
- Predicted action
- Confidence score

---

## 🔍 Methodology

### 1. Frame Extraction
Video frames are extracted using OpenCV.

### 2. Frame Preprocessing
Each frame is:
- Resized to 224×224
- Converted to RGB
- Normalized using X-CLIP mean and standard deviation values

### 3. Temporal Windowing
Videos are divided into overlapping clips using:

```text
Window Size = 8 Frames
Stride = 4 Frames
```

This enables temporal understanding of actions across consecutive frames.

### 4. Zero-Shot Classification
Each clip is compared against text descriptions of actions using X-CLIP's vision-language embeddings.

### 5. Prediction Generation
The action with the highest probability score is selected and displayed along with confidence.

---

## 📈 Applications

- Human Activity Recognition
- Smart Surveillance Systems
- Sports Analytics
- Video Content Tagging
- Industrial Safety Monitoring
- Human-Robot Interaction
- Intelligent Video Search

---

## 🔮 Future Improvements

- Real-time webcam inference
- Custom action classes
- Multi-person action recognition
- Action detection with bounding boxes
- Fine-tuning on domain-specific datasets
- Web dashboard for live predictions
- Integration with robotics systems

---

## 🎯 Key Learning Outcomes

- Transformer-based video understanding
- Vision-language models
- Zero-shot learning
- Temporal feature extraction
- Video preprocessing pipelines
- Hugging Face model deployment

---

⭐ If you found this project useful, consider starring the repository.
