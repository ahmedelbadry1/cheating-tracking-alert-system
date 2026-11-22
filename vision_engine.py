import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

SCREENSHOT_DIR = Path("alerts") / "screenshots"


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / (area_a + area_b - inter)


def _center_inside(inner_box, outer_box):
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


class ExamMonitorEngine:
    def __init__(self, custom_weights="best.pt", conf_threshold=0.35, cooldown_sec=5.0):
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.cooldown_sec = cooldown_sec
        self.person_model = YOLO("yolov8s.pt")
        self.cheating_model = YOLO(custom_weights) if os.path.exists(custom_weights) else None
        self.last_alert_time = {}
        self.cheat_streak = {}

    def _format_video_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def process_video(self, video_path, frame_callback, alert_callback, should_stop):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        self.last_alert_time.clear()
        self.cheat_streak.clear()

        try:
            while cap.isOpened() and not should_stop():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                video_sec = frame_idx / fps

                # 1. Track students with YOLOv8s + BoT-SORT
                person_res = self.person_model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False,
                )[0]

                # 2. Detect phones with COCO YOLOv8s (to verify phone alerts)
                phone_res = self.person_model.predict(
                    frame,
                    classes=[67],
                    conf=0.25,
                    verbose=False,
                )[0]
                phone_boxes = []
                if phone_res.boxes is not None and len(phone_res.boxes) > 0:
                    for b, c in zip(
                        phone_res.boxes.xyxy.cpu().numpy(),
                        phone_res.boxes.conf.cpu().numpy(),
                    ):
                        phone_boxes.append((b, float(c)))

                # 3. Detect cheating behaviors with trained best.pt
                cheat_boxes = []
                if self.cheating_model is not None:
                    cheat_res = self.cheating_model.predict(
                        frame,
                        conf=self.conf_threshold,
                        verbose=False,
                    )[0]
                    if cheat_res.boxes is not None and len(cheat_res.boxes) > 0:
                        for b, cls_id, c in zip(
                            cheat_res.boxes.xyxy.cpu().numpy(),
                            cheat_res.boxes.cls.cpu().numpy().astype(int),
                            cheat_res.boxes.conf.cpu().numpy(),
                        ):
                            lbl = cheat_res.names[cls_id].lower()
                            if lbl != "normal_action":
                                cheat_boxes.append((b, lbl, float(c)))

                annotated = frame.copy()
                boxes = person_res.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy_list = boxes.xyxy.cpu().numpy()
                    conf_list = boxes.conf.cpu().numpy()
                    id_list = (
                        boxes.id.cpu().numpy().astype(int)
                        if boxes.id is not None
                        else list(range(1, len(xyxy_list) + 1))
                    )

                    # Assign each cheating box to ONLY the single best-matching student
                    student_alerts = {}
                    for c_box, c_lbl, c_conf in cheat_boxes:
                        best_idx = -1
                        best_score = 0.0
                        for idx, p_box in enumerate(xyxy_list):
                            score = _iou(p_box, c_box)
                            if _center_inside(c_box, p_box):
                                score += 0.25
                            if score > best_score:
                                best_score = score
                                best_idx = idx

                        if best_idx != -1 and best_score > 0.25:
                            # Require COCO phone agreement for phone classes (unless very high conf)
                            if "phone" in c_lbl and c_conf < 0.55:
                                p_box = xyxy_list[best_idx]
                                has_coco_phone = any(
                                    _center_inside(ph_b, p_box) or _iou(p_box, ph_b) > 0.05
                                    for ph_b, _ in phone_boxes
                                )
                                if not has_coco_phone:
                                    continue

                            label_clean = c_lbl.replace("_", " ").title()
                            if best_idx not in student_alerts or c_conf > student_alerts[best_idx][1]:
                                student_alerts[best_idx] = (label_clean, c_conf)

                    # Also match verified COCO phones directly to the single best student
                    for ph_b, ph_c in phone_boxes:
                        best_idx = -1
                        best_score = 0.0
                        for idx, p_box in enumerate(xyxy_list):
                            if _center_inside(ph_b, p_box):
                                score = _iou(p_box, ph_b) + 0.5
                                if score > best_score:
                                    best_score = score
                                    best_idx = idx
                        if best_idx != -1 and best_idx not in student_alerts:
                            student_alerts[best_idx] = ("Phone Detected", ph_c)

                    # Draw boxes and trigger alerts after 3 consecutive frames
                    for idx, (box, conf, track_id) in enumerate(zip(xyxy_list, conf_list, id_list)):
                        x1, y1, x2, y2 = map(int, box)
                        tid = int(track_id)

                        if idx in student_alerts:
                            behavior, event_conf = student_alerts[idx]
                            self.cheat_streak[tid] = self.cheat_streak.get(tid, 0) + 1
                            is_cheating = self.cheat_streak[tid] >= 3
                        else:
                            self.cheat_streak[tid] = 0
                            behavior = "Normal"
                            event_conf = float(conf)
                            is_cheating = False

                        color = (30, 30, 230) if is_cheating else (40, 200, 60)
                        tag = f"ID {tid} | {behavior} ({event_conf:.2f})"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        label_y = y1 - 6 if y1 > 26 else y1 + 18
                        bg_y1 = max(0, label_y - 18)
                        bg_y2 = label_y + 4
                        cv2.rectangle(annotated, (x1, bg_y1), (x1 + len(tag) * 9, bg_y2), color, -1)
                        cv2.putText(
                            annotated,
                            tag,
                            (x1 + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        if is_cheating:
                            now_ts = time.time()
                            last_ts = self.last_alert_time.get(tid, 0.0)
                            if now_ts - last_ts >= self.cooldown_sec:
                                self.last_alert_time[tid] = now_ts
                                dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                shot_path = str(SCREENSHOT_DIR / f"alert_ID{tid}_{file_stamp}.jpg")
                                cv2.imwrite(shot_path, annotated)

                                alert_callback(
                                    {
                                        "timestamp": dt_str,
                                        "student_id": tid,
                                        "behavior": behavior,
                                        "confidence": round(event_conf, 3),
                                        "video_time": self._format_video_time(video_sec),
                                        "screenshot_path": shot_path,
                                    }
                                )

                frame_callback(annotated)
        finally:
            cap.release()