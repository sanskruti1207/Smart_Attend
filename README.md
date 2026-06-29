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

