# 🎓 Cheating Tracking & Alert System

An end-to-end Computer Vision desktop application for monitoring exam videos, tracking students in real time, detecting suspicious cheating behaviors, logging incidents to a local database, and triggering instant alerts.

Built as **Project 3/3 (Sprint 2)** for the **AI & Data Analytics internship** at **Instant Software Solutions**.

---

## ✨ Key Features
- **Real-Time Student Tracking:** Tracks every student with a persistent ID using **YOLOv8s + BoT-SORT**.
- **Cheating Behavior Detection:** Custom fine-tuned **YOLOv8s (`best.pt`)** trained on a **4,183-image** Roboflow dataset covering **11 classes** (`normal_action`, `look_around`, `using_phone`, `exchanging`, `using_document`, `using_earphone`, `using_smartwatch`, `phone`, `document`, `earphone`, `smartwatch`).
- **False-Alarm Reduction:**
  - Dual-model verification using COCO **YOLOv8s** (`cell phone`) to confirm phone-related alerts.
  - 3-frame consecutive detection filter and a 5-second per-student alert cooldown.
- **Visual Indicators:** Draws **green** bounding boxes for normal students and switches to **red** when cheating is detected.
- **Desktop GUI (Tkinter — Dark Mode):**
  - Upload exam videos (`MP4`, `AVI`, `MOV`, `MKV`) with live playback and Stop controls.
  - Real-time **Live Cheating Alerts** feed and instant **Pop-up Toast** notifications.
  - Custom application icon.
- **Database & Evidence Logging (SQLite):**
  - Automatically captures and saves annotated screenshots locally in `alerts/screenshots/`.
  - Logs `Date/Time`, `Student ID`, `Behavior`, `Confidence`, `Video Time`, and `Screenshot Path` into `alerts/cheating_alerts.db`.
  - Built-in **Alert History** table with an interactive **Screenshot Viewer**.

---

## 📂 Project Structure
```text
cheating-tracking-alert-system/
├── app.py                         # Tkinter GUI, live feed, toast alerts & history viewer
├── vision_engine.py               # YOLOv8s + BoT-SORT tracking & cheating detection engine
├── db_manager.py                  # SQLite database manager for alert records
├── best.pt                        # Custom fine-tuned YOLOv8s weights
├── cheating_yolov8s_colab.ipynb   # Google Colab training & evaluation notebook
└── requirements.txt               # Python dependencies
