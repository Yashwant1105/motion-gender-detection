# 🎥 Motion & Gender Detection - Execution Guide

## 📦 Run Instructions

### ▶️ Webcam Mode

python motion_gender_detection.py --mode webcam --process sequential  
python motion_gender_detection.py --mode webcam --process parallel  
python motion_gender_detection.py --mode webcam --process parallel --save  

---

### 🎞️ Video Mode

python motion_gender_detection.py --mode video --file path/to/video.mp4 --process sequential  
python motion_gender_detection.py --mode video --file path/to/video.mp4 --process parallel  
python motion_gender_detection.py --mode video --file path/to/video.mp4 --process parallel --save  

---

### 🖼️ Image Mode

python motion_gender_detection.py --mode image --file path/to/image.jpg  
python motion_gender_detection.py --mode image --file path/to/image.jpg --save  

---

### 🛠️ Help

python motion_gender_detection.py --help
