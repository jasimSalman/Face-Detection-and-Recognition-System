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
        self.root.geometry("600x400")
        self.root.config(bg="#F0F0F0")

        self.main_menu = None
        # self.main_menu.pack(fill="both", expand=True)

        self.image_recognition_frame = tk.Frame(root)        
        self.realtime_recognition = RealtimeRecognition(self)
        
        self.create_main_menu()

        self.image_list = []

    def create_main_menu(self):
        self.main_menu = tk.Frame(root, bg="#F0F0F0")
        self.main_menu.pack(fill="both", expand=True)
        tk.Label(self.main_menu, text="Face Detection and Recognition", font=("Helvetica", 24, "bold"), bg="#F0F0F0", fg="#333").pack(pady=20)
        self.create_button(self.main_menu, "Real-Time Recognition", self.show_real_time)
        self.create_button(self.main_menu, "Image Recognition", self.show_image_recognition)

    def switch_to_main_menu(self):
        for widget in self.realtime_recognition.real_time_frame.winfo_children():
            widget.destroy()
        self.realtime_recognition.real_time_frame.pack_forget()
        self.create_main_menu()

    def create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, font=("Helvetica", 14), bg="#4CAF50", fg="white", relief="flat", width=20, height=2, command=command)
        button.pack(pady=15)
        button.config(activebackground="#45a049", activeforeground="white")

    def show_real_time(self):
        self.realtime_recognition.show_real_time()

    def show_image_recognition(self):
        self.switch_frame(self.image_recognition_frame)
        self.setup_image_recognition_ui()

    def switch_frame(self, frame):
        # Hide the current frame if it's visible
        for widget in self.root.main_menu.winfo_children():
            widget.pack_forget()

        # Now pack the new frame
        frame.pack(fill="both", expand=True)

    def setup_image_recognition_ui(self):
        self.create_button(self.image_recognition_frame, "Upload Images", self.upload_images)
        self.image_container = tk.Frame(self.image_recognition_frame, bg="#F0F0F0")
        self.image_container.pack(pady=20)
        self.create_button(self.image_recognition_frame, "Start Recognition", self.recognize_images)
        self.create_button(self.image_recognition_frame, "Back to Menu", lambda: self.switch_frame(self.main_menu))

    def upload_images(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        for filepath in filepaths:
            img = Image.open(filepath)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.image_container, image=img, bg="#F0F0F0")
            lbl.image = img
            lbl.pack(side=tk.LEFT, padx=5, pady=5)
            self.image_list.append(filepath)

    def recognize_images(self):
        print("Starting recognition for uploaded images...")
        for img_path in self.image_list:
            print(f"Recognizing: {img_path}")


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    root.mainloop()
