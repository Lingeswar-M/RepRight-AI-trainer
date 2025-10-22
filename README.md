# RepRight AI Trainer

An AI-powered fitness trainer that uses computer vision to count exercise repetitions and provide real-time form feedback for dumbbell exercises.

## ✨ Features

- 🧠 **Automatic Exercise Detection**: Intelligently detects whether you're doing bicep curls or shoulder press
- 🏋️‍♂️ **Real-Time Rep Counting**: Accurately tracks your repetitions using pose estimation
- 💪 **Form Feedback**: Provides instant visual feedback on your exercise form
- 📹 **Video Input Support**: Works with both live webcam and pre-recorded videos
- 📊 **Session Recording**: Automatically saves analyzed workout videos

## 🚀 How It Works

The application uses:
- **OpenCV** for video capture and processing
- **MediaPipe** for real-time pose detection and landmark tracking
- **NumPy** for angle calculations and mathematical operations

## 📋 Requirements
- **opencv-python**
- **mediapipe**
- **numpy**
