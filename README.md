# SmartAttend

SmartAttend is a production-ready AI Smart Attendance Platform. It combines face recognition, real-time liveness checks (anti-spoofing), session authentication, and machine learning analytics to streamline attendance management.

---

## 🌟 Key Features

* **Role-Based Authentication**: Secure login portals for both Students and Teachers.
* **Liveness Detection**: Anti-spoofing checks (Smiles, Head turns) utilizing MediaPipe Face Mesh to prevent photo or video spoofing.
* **Lighting-Invariant Facial Recognition**: Uses OpenCV YuNet (Face Detection) and SFace (Face Embedding) with Cosine Similarity matching.
* **ML Attendance Prediction**: Decision Tree models analyze student histories to forecast attendance risks (Safe, Warning, Critical).
* **Flexible Database Logging**: Relational SQLite storage prevents duplicate daily markings and maintains comprehensive system audit logs.

---

## 🔑 Test Credentials

| Portal | Username | Password | Access / Features |
| :--- | :--- | :--- | :--- |
| **Teacher Portal** | `teacher` | `teacher123` | Analytics dashboard, registration, record editor, audit logs. |
| **Student Portal** | `student` | `student123` | Mark attendance via camera, view personal attendance rates & predictions. |

---

## 📁 Project Structure

```text
SmartAttend/
├── src/
│   ├── anti_spoofing/   # MediaPipe liveness detection challenges (liveness.py)
│   ├── attendance/      # Attendance marking & punctuality rules (attendance_marker.py)
│   ├── database_helper/ # SQLite CRUD transactions (database_helper.py)
│   ├── prediction/      # ML risk predictor model (predictor.py)
│   ├── recognition/     # Face detection & SVM classification (face_classifier.py)
│   └── registration/    # Dataset collection & training (collector.py, manager.py)
├── database/            # SQLite database files
├── dataset/             # Registered student face image crops
├── models/              # Classifier models and ONNX neural network weights
├── attendance/          # Exported CSV and Excel reports
├── app.py               # Main Streamlit web application
├── verify_system.py     # Automated test suite
└── requirements.txt     # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have **Python 3.13** installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the Streamlit application:
```bash
streamlit run app.py
```

### 3. Open the App
The application will launch on your default browser at:
**`http://localhost:8501`**

---

## 🧪 Verification
To verify the system integrity and run unit tests, execute:
```bash
python verify_system.py
```