# 👁️ Eye & Nose Controlled Automation System (UiPath + Python)

## 📌 Project Overview

This project is an **AI-based Human-Computer Interaction system** developed using **UiPath and Python**, which enables users to control the computer **cursor, clicks, and scrolling using facial movements and eye blinks**.

The system leverages **computer vision and machine learning techniques** to track the user's **nose position and eye aspect ratio (EAR)** for gesture-based control.

---

## 🚀 Features

* 🎯 Cursor movement using **nose tracking**
* 👁️ Blink detection for interaction:

  * **Single Blink → Mouse Click**
  * **Double Blink → Scroll Down**
* 📏 Automatic **face distance calibration**
* 🧠 Smart **Kalman Filter smoothing** for stable cursor movement
* 🎥 Real-time **face landmark detection using MediaPipe**
* 🖥️ Fullscreen gaze tracking interface
* 🔁 Recalibration option during runtime

---

## 🏗️ System Architecture

### 🔹 UiPath

* Acts as the **automation controller**
* Integrates with Python script
* Handles workflow execution via `.xaml` file

### 🔹 Python Module

* Performs:

  * Face detection
  * Nose tracking
  * Blink detection
  * Cursor control
* Uses libraries like:

  * OpenCV
  * MediaPipe
  * NumPy
  * PyAutoGUI

---

## 📂 Project Structure

```
EyesystemController/
│
├── Main.xaml                # UiPath workflow
├── script/
│   ├── pupil.py            # Core Python logic
│   ├── calibration_nose.npz
│   └── shape_predictor_68_face_landmarks.dat
│
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies Used

* 🤖 UiPath Studio
* 🐍 Python 3.x
* 📸 OpenCV
* 🧠 MediaPipe Face Mesh
* 🔢 NumPy & SciPy
* 🖱️ PyAutoGUI

---

## 🧪 How It Works

1. **Camera Initialization**

   * Captures real-time video feed

2. **Face & Landmark Detection**

   * Detects eyes and nose using MediaPipe

3. **Calibration (3 Steps)**

   * Face positioning
   * Nose rest position detection
   * 9-point gaze calibration

4. **Tracking Phase**

   * Nose movement → Cursor movement
   * Eye blinking → Action triggers

---

## 🎮 Controls

| Action      | Gesture            |
| ----------- | ------------------ |
| Move Cursor | Head/Nose Movement |
| Left Click  | Single Blink       |
| Scroll Down | Double Blink       |
| Recalibrate | Press `R`          |
| Exit        | Press `Q`          |

---

## ▶️ How to Run

### 🔹 Step 1: Install Dependencies

```bash
pip install opencv-python mediapipe numpy pyautogui scipy
```

### 🔹 Step 2: Open UiPath

* Open `Main.xaml` in UiPath Studio
* Configure Python path in UiPath

### 🔹 Step 3: Run the Project

* Execute workflow from UiPath
* Camera will start automatically
* Follow on-screen calibration steps

---

## ⚠️ Requirements

* Webcam enabled device
* Good lighting conditions
* Windows OS (for full functionality)

---

## 🧠 Future Enhancements

* Voice control integration 🎤
* Gesture-based zoom and drag ✋
* Mobile support 📱
* AI-based fatigue detection 😴

---

## 👨‍💻 Author

**Akram Ali Faridi**

---

## 📜 License

This project is for educational and research purposes.
