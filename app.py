import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

from real_time_recognition import RealtimeRecognition 


class RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection and Recognition System")

        self.realtime_recognition = RealtimeRecognition(self.root)

        self.main_menu = tk.Frame(root)
        self.main_menu.pack()

        tk.Label(self.main_menu, text="Choose Mode", font=("Arial", 20)).pack(pady=10)
        tk.Button(self.main_menu, text="Real-Time Recognition", width=20, command=self.show_real_time).pack(pady=10)
        tk.Button(self.main_menu, text="Image Recognition", width=20, command=self.show_image_recognition).pack(pady=10)

        self.real_time_frame = tk.Frame(root)
        self.image_recognition_frame = tk.Frame(root)
        
        self.image_list = []

    def show_real_time(self):
        self.realtime_recognition.show_real_time()

    def show_image_recognition(self):
        self.switch_frame(self.image_recognition_frame)
        self.setup_image_recognition_ui()

    def switch_frame(self, frame):
        if self.realtime_recognition.current_frame:
            self.realtime_recognition.current_frame.pack_forget()
        self.realtime_recognition.current_frame = frame
        frame.pack()

    def setup_image_recognition_ui(self):
        tk.Button(self.image_recognition_frame, text="Upload Images", command=self.upload_images).pack(pady=10)
        self.image_container = tk.Frame(self.image_recognition_frame)
        self.image_container.pack(pady=10)
        tk.Button(self.image_recognition_frame, text="Start Recognition", command=self.recognize_images).pack(pady=10)
        tk.Button(self.image_recognition_frame, text="Back to Menu", command=lambda: self.switch_frame(self.main_menu)).pack(pady=10)

    def upload_images(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        for filepath in filepaths:
            img = Image.open(filepath)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.image_container, image=img)
            lbl.image = img
            lbl.pack(side=tk.LEFT, padx=5, pady=5)
            self.image_list.append(filepath)

    def recognize_images(self):
        # Implement image recognition logic here
        print("Starting recognition for uploaded images...")
        for img_path in self.image_list:
            print(f"Recognizing: {img_path}")


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    root.mainloop()
