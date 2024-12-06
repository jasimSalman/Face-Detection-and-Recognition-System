import cv2
import os
import numpy as np
import json
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox



class RealtimeRecognition:
    def __init__(self, root):
        self.root = root
        self.real_time_frame = tk.Frame(root)
        self.current_frame = None
        self.face_info_saved = False
        self.face_id = None  
        self.face_name = None

        self.images_dir = './images/'
        self.cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
        self.names_json_filename = 'names.json'
        self.trainer_filename = 'trainer.yml'
        self.count = 0

        self.create_directory(self.images_dir)
        self.camera_label = None
        self.cap = None
        self.running = False

    # Function to create directory if not exists
    def create_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_face_id(self, directory: str) -> int:
        user_ids = []
        for filename in os.listdir(directory):
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
        user_ids = sorted(list(set(user_ids)))
        max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
        return max_user_ids

    def save_name(self,face_id: int, face_name: str, filename: str) -> None:
        """Saves the face ID and corresponding name to a JSON file."""
        names_json = {}

        # If the file does not exist, create an empty one
        if not os.path.exists(filename):
            with open(filename, 'w') as fs:
                json.dump(names_json, fs, ensure_ascii=False, indent=4)

        # Now try to load it
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as fs:
                try:
                    names_json = json.load(fs)
                except json.JSONDecodeError:
                    print(f"Error: {filename} contains invalid JSON. Starting with an empty dictionary.")
                    names_json = {}

        # Update the JSON with the new face ID and name
        names_json[str(face_id)] = face_name

        # Save the updated JSON back to the file
        with open(filename, 'w') as fs:
            json.dump(names_json, fs, ensure_ascii=False, indent=4)

    def show_real_time(self):
        self.switch_frame(self.real_time_frame)
        self.setup_real_time_ui()

    def show_image_recognition(self):
        self.switch_frame(self.image_recognition_frame)
        self.setup_image_recognition_ui()

    def switch_frame(self, frame):
        if self.current_frame:
            self.current_frame.pack_forget()
        self.current_frame = frame
        frame.pack()

    def setup_real_time_ui(self):
        if not self.camera_label:
            self.camera_label = tk.Label(self.real_time_frame)
            self.camera_label.pack()

        tk.Label(self.real_time_frame, text="Enter Name:", font=("Arial", 14)).pack(pady=10)
        self.name_entry = tk.Entry(self.real_time_frame, font=("Arial", 14))
        self.name_entry.pack(pady=5)

        tk.Button(self.real_time_frame, text="Start Detection", command=self.start_camera).pack(pady=10)
        tk.Button(self.real_time_frame, text="Recognize", command=self.start_recognition).pack(pady=10)
        tk.Button(self.real_time_frame, text="Back to Menu", command=lambda: self.stop_camera(self.main_menu)).pack(pady=10)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_camera()

    def stop_camera(self, next_frame):
        if self.cap:
            self.running = False
            self.cap.release()
            self.camera_label.config(image="")
        self.switch_frame(next_frame)

    def update_camera(self):
        # If face ID and name are not saved yet
        if not self.face_info_saved:
            # Get face ID and name only once
            self.face_cascade = cv2.CascadeClassifier(self.cascade_classifier_filename)
            self.face_id = self.get_face_id(self.images_dir)
            self.face_name = self.name_entry.get()
            self.save_name(self.face_id, self.face_name, self.names_json_filename)
            self.face_info_saved = True  # Mark as saved



        if self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    self.count += 1
                    face_region = gray[y:y + h, x:x + w]
                    face_path = f'./images/Users-{self.face_id}-{self.count}.jpg'
                    cv2.imwrite(face_path, face_region)

                # Stop capturing after 30 images
                if self.count >= 30:
                    self.running = False
                    self.cap.release()
                    cv2.destroyAllWindows()
                    self.on_capture_complete()
                    return

                # Display frame in Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400, 300))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.camera_label.config(image=img)
                self.camera_label.image = img
            
            # Continue the loop after 10 ms
            self.camera_label.after(10, self.update_camera)

    def on_capture_complete(self):
        """Callback when the face capture process is complete."""
        messagebox.showinfo("Info", f"Captured {self.count} images successfully!")

            # Automatically train the face recognizer
        try:
            self.train_face_recognizer('./images/')
            messagebox.showinfo("Info", "Face recognizer training completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error during training: {str(e)}")
            
        self.stop_camera(self.real_time_frame)

    def train_face_recognizer(self, path: str):
        """Trains the face recognizer after capturing faces."""
        print("\n[INFO] Training face recognizer...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(self.cascade_classifier_filename)

        def get_images_and_labels(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path)]
            face_samples = []
            ids = []
            for image_path in image_paths:
                PIL_img = Image.open(image_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                face_id = int(os.path.split(image_path)[-1].split("-")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(face_id)
            return face_samples, ids

        faces, ids = get_images_and_labels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write(self.trainer_filename)
        print(f"\n[INFO] {len(np.unique(ids))} faces trained.")

    def load_resources(self):
        """Load resources (trainer, cascade, and names)."""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(self.cascade_classifier_filename)
        self.recognizer.read(self.trainer_filename)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        with open(self.names_json_filename, 'r') as fs:
            names = json.load(fs)
            self.names = list(names.values())

    def start_recognition(self):
        """Start face recognition in a separate thread."""
        self.running = True
        recognition_thread = threading.Thread(target=self.recognize_real_time)
        recognition_thread.daemon = True  # Allow thread to exit when the main app exits
        recognition_thread.start()

    def recognize_real_time(self):
        """Real-time face recognition loop."""
        self.load_resources()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not access the camera.")
            return

        while self.running:
            ret, img = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                if id < len(self.names) and confidence < 50:
                    name = self.names[id]
                else:
                    name = "Unknown"
                
                confidence_text = f"  {round(confidence)}%"
                cv2.putText(img, name, (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

            # Convert OpenCV image (BGR) to Tkinter-compatible image (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 300))
            img = ImageTk.PhotoImage(Image.fromarray(img))

            # Update the Tkinter label with the new frame
            if self.camera_label:
                self.camera_label.config(image=img)
                self.camera_label.image = img

            # Break the loop if user presses ESC (key code 27)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False

        self.stop_recognition()

    def stop_recognition(self):
        """Stop the face recognition process."""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

