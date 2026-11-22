import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk

from db_manager import AlertDatabase
from vision_engine import ExamMonitorEngine

ICON_FILE = "app_icon.png"


def create_app_icon():
    if not os.path.exists(ICON_FILE):
        img = Image.new("RGBA", (64, 64), (24, 28, 36, 255))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((4, 4, 60, 60), radius=12, outline=(235, 64, 52, 255), width=4)
        draw.ellipse((20, 20, 44, 44), outline=(255, 255, 255, 255), width=3)
        draw.ellipse((28, 28, 36, 36), fill=(235, 64, 52, 255))
        img.save(ICON_FILE)


class CheatingMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exam Cheating Tracking & Alert System")
        self.geometry("1180x700")
        self.configure(bg="#12151c")
        create_app_icon()
        self.iconphoto(True, tk.PhotoImage(file=ICON_FILE))

        self.db = AlertDatabase()
        self.engine = None
        self.stop_requested = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.event_queue = queue.Queue()

        self._build_ui()
        self.after(30, self._update_loop)

    def _build_ui(self):
        top_bar = tk.Frame(self, bg="#1c212b", pady=10, padx=12)
        top_bar.pack(fill="x")

        btn_style = {
            "font": ("Segoe UI", 10, "bold"),
            "fg": "white",
            "bd": 0,
            "padx": 14,
            "pady": 6,
            "cursor": "hand2",
        }
        self.btn_upload = tk.Button(
            top_bar, text="Upload Video", bg="#2563eb", command=self.on_upload, **btn_style
        )
        self.btn_upload.pack(side="left", padx=(0, 8))

        self.btn_stop = tk.Button(
            top_bar, text="Stop", bg="#dc2626", command=self.on_stop, state="disabled", **btn_style
        )
        self.btn_stop.pack(side="left", padx=(0, 8))

        tk.Button(
            top_bar, text="Alert History", bg="#374151", command=self.open_history, **btn_style
        ).pack(side="left")

        self.status_lbl = tk.Label(
            top_bar, text="Status: Ready", bg="#1c212b", fg="#9ca3af", font=("Segoe UI", 10)
        )
        self.status_lbl.pack(side="right")

        content = tk.Frame(self, bg="#12151c", padx=12, pady=10)
        content.pack(fill="both", expand=True)

        self.video_panel = tk.Label(
            content,
            bg="#090b0f",
            text="Upload an exam video to start monitoring",
            fg="#6b7280",
            font=("Segoe UI", 12),
        )
        self.video_panel.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(content, bg="#1c212b", width=330, padx=10, pady=10)
        sidebar.pack(side="right", fill="y", padx=(12, 0))
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Live Cheating Alerts", bg="#1c212b", fg="#f87171", font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.alert_list = tk.Listbox(
            sidebar,
            bg="#12151c",
            fg="#f3f4f6",
            selectbackground="#2563eb",
            font=("Consolas", 9),
            bd=0,
            highlightthickness=0,
        )
        self.alert_list.pack(fill="both", expand=True)

    def on_upload(self):
        video_path = filedialog.askopenfilename(
            title="Select Exam Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not video_path:
            return
        self.stop_requested = False
        self.btn_upload.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Loading YOLO model & processing...")
        threading.Thread(target=self._worker, args=(video_path,), daemon=True).start()

    def _worker(self, video_path):
        try:
            if self.engine is None:
                self.engine = ExamMonitorEngine()

            def push_frame(frame):
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

            def push_alert(record):
                self.db.insert_alert(record)
                self.event_queue.put(("alert", record))

            self.engine.process_video(video_path, push_frame, push_alert, lambda: self.stop_requested)
        except Exception as exc:
            self.event_queue.put(("error", str(exc)))
        finally:
            self.event_queue.put(("done", None))

    def on_stop(self):
        self.stop_requested = True
        self.status_lbl.config(text="Status: Stopping...")

    def _update_loop(self):
        try:
            frame = self.frame_queue.get_nowait()
            h, w = frame.shape[:2]
            pw = max(self.video_panel.winfo_width(), 400)
            ph = max(self.video_panel.winfo_height(), 300)
            scale = min(pw / w, ph / h)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_panel.config(image=photo, text="")
            self.video_panel.image = photo
        except queue.Empty:
            pass

        while not self.event_queue.empty():
            kind, payload = self.event_queue.get()
            if kind == "error":
                messagebox.showerror("Processing Error", payload)
            elif kind == "done":
                self.btn_upload.config(state="normal")
                self.btn_stop.config(state="disabled")
                self.status_lbl.config(text="Status: Finished")
            elif kind == "alert":
                line = f"[{payload['video_time']}] ID {payload['student_id']} - {payload['behavior']} ({payload['confidence']})"
                self.alert_list.insert(0, line)
                self._show_toast(payload)

        self.after(30, self._update_loop)

    def _show_toast(self, record):
        toast = tk.Toplevel(self)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(bg="#b91c1c")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        toast.geometry(f"330x125+{sw - 350}+{sh - 180}")
        msg = (
            "CHEATING ALERT DETECTED\n\n"
            f"Behavior: {record['behavior']}\n"
            f"Student ID: {record['student_id']} | Time: {record['timestamp'][11:16]}\n"
            "Screenshot saved successfully."
        )
        tk.Label(
            toast, text=msg, bg="#b91c1c", fg="white", justify="left", font=("Segoe UI", 9, "bold"), padx=12, pady=10
        ).pack(anchor="w")
        toast.after(4500, toast.destroy)

    def open_history(self):
        win = tk.Toplevel(self)
        win.title("Cheating Alert History")
        win.geometry("920x440")
        win.configure(bg="#12151c")

        cols = ("Date / Time", "Student ID", "Behavior", "Confidence", "Video Time", "Screenshot Path")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=260 if col == "Screenshot Path" else 120)

        for row in self.db.fetch_all():
            tree.insert("", "end", values=row)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def open_shot(_=None):
            selected = tree.selection()
            if not selected:
                return
            shot_path = tree.item(selected[0])["values"][5]
            if not os.path.exists(shot_path):
                messagebox.showwarning("Not Found", "Screenshot file was not found.")
                return
            viewer = tk.Toplevel(win)
            viewer.title(os.path.basename(shot_path))
            img = Image.open(shot_path)
            img.thumbnail((960, 680))
            ph = ImageTk.PhotoImage(img)
            lbl = tk.Label(viewer, image=ph)
            lbl.image = ph
            lbl.pack()

        tree.bind("<Double-1>", open_shot)
        tk.Button(
            win,
            text="View Selected Screenshot",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=14,
            pady=6,
            command=open_shot,
        ).pack(pady=(0, 10))


if __name__ == "__main__":
    CheatingMonitorApp().mainloop()
