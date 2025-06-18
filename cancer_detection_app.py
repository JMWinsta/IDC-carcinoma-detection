import os
import sys
import io
import json
import logging
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

# Configure logging
logging.basicConfig(filename="cancer_app.log", level=logging.INFO, 
                   format="%(asctime)s - %(levelname)s - %(message)s")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class CancerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Cell Detection")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f5")
        self.root.resizable(False, False)
        
        # Load configuration
        self.config = self.load_config()
        
        # Set app icon
        try:
            self.root.iconbitmap("app_icon.ico")
        except:
            pass
        
        # Initialize variables
        self.image_path = None
        self.model = None
        self.model_loaded = False
        self.prediction_made = False
        self.confidence_scores = None
        
        # Load the model
        self.load_model_file()
        
        # Create the UI
        self.create_ui()
        
    def load_config(self):
        default_config = {
            "model_path": "cancer_detection_model.h5",
            "target_size": [48, 48],
            "threshold": 0.5
        }
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            with open("config.json", "w") as f:
                json.dump(default_config, f)
            return default_config
    
    def load_model_file(self):
        try:
            # Use resource_path() to get the bundled model's location
            model_path = resource_path(self.config["model_path"])
            self.model = load_model(model_path)
            self.model_loaded = True
            logging.info("Model loaded successfully from default path")
        except FileNotFoundError:
            logging.warning("Default model not found, prompting user selection")
            file_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("HDF5 files", "*.h5")]
            )
            if file_path:
                self.model = load_model(file_path)
                self.model_loaded = True
                self.config["model_path"] = file_path
                with open("config.json", "w") as f:
                    json.dump(self.config, f)
                logging.info(f"Model loaded from user-selected path: {file_path}")
        except Exception as e:
            self.model_loaded = False
            logging.error(f"Failed to load model: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
    
    def create_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=70)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="Cancer Cell Detection System", 
                             font=("Helvetica", 20, "bold"), fg="white", bg="#2c3e50")
        title_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#f0f0f5")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel
        left_panel = tk.Frame(content_frame, bg="#f0f0f5", width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_frame = tk.Frame(left_panel, bg="white", height=350, width=380)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.image_frame.pack_propagate(False)
        
        self.placeholder_label = tk.Label(self.image_frame, text="No Image Selected", 
                                       font=("Helvetica", 14), bg="white")
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg="#f0f0f5")
        button_frame.pack(fill=tk.X, pady=10)
        
        upload_button = ttk.Button(button_frame, text="Upload Image", 
                                 command=self.upload_image, style="Accent.TButton")
        upload_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.predict_button = ttk.Button(button_frame, text="Detect Cancer", 
                                       command=self.predict_cancer, state="disabled",
                                       style="Accent.TButton")
        self.predict_button.pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Please upload an image to begin")
        status_label = tk.Label(left_panel, textvariable=self.status_var,
                              font=("Helvetica", 10), bg="#f0f0f5", fg="#555555")
        status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Right panel with tabs
        right_panel = tk.Frame(content_frame, bg="#f0f0f5", width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.tab_control = ttk.Notebook(right_panel)
        self.results_tab = ttk.Frame(self.tab_control)
        self.metrics_tab = ttk.Frame(self.tab_control)
        self.history_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.results_tab, text="Results")
        self.tab_control.add(self.metrics_tab, text="Metrics")
        self.tab_control.add(self.history_tab, text="History")
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        self.create_results_tab()
        self.create_metrics_tab()
        self.create_history_tab()
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#2c3e50", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_label = tk.Label(footer_frame, text="© 2025 Cancer Detection System", 
                              font=("Helvetica", 8), fg="white", bg="#2c3e50")
        footer_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Style
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Helvetica", 11))
        
        if not self.model_loaded:
            messagebox.showwarning("Model Not Found", 
                                 "Please ensure a valid model file is available.")
    
    def create_results_tab(self):
        results_frame = tk.Frame(self.results_tab, padx=15, pady=15)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Diagnosis Results", 
                font=("Helvetica", 14, "bold"), pady=10).pack()
        ttk.Separator(results_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.result_label = tk.Label(results_frame, text="No prediction yet", 
                                   font=("Helvetica", 18, "bold"), wraplength=350,
                                   justify=tk.CENTER)
        self.result_label.pack(pady=20)
        
        confidence_frame = tk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, pady=10)
        
        self.confidence_label = tk.Label(confidence_frame, text="Confidence: --",
                                       font=("Helvetica", 12))
        self.confidence_label.pack()
        
        self.confidence_bar = ttk.Progressbar(confidence_frame, orient="horizontal", 
                                            length=300, mode="determinate")
        self.confidence_bar.pack(pady=10)
        
        image_info_frame = tk.Frame(results_frame, pady=10)
        image_info_frame.pack(fill=tk.X)
        
        self.image_info_label = tk.Label(image_info_frame, text="Image: None",
                                       font=("Helvetica", 10), justify=tk.LEFT)
        self.image_info_label.pack(anchor=tk.W)
    
    def create_metrics_tab(self):
        metrics_frame = tk.Frame(self.metrics_tab, padx=15, pady=15)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(metrics_frame, text="Detailed Metrics", 
                font=("Helvetica", 14, "bold"), pady=10).pack()
        ttk.Separator(metrics_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.fig1, self.ax1 = plt.subplots(figsize=(5, 2))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=metrics_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.X, pady=10)
        
        self.ax1.set_title("Confidence Scores")
        self.ax1.set_xlim(0, 1)
        self.ax1.set_yticks([0, 1])
        self.ax1.set_yticklabels(["Non-Cancerous", "Cancerous"])
        self.ax1.set_xlabel("Confidence Score")
        self.fig1.tight_layout()
        
        detailed_metrics_frame = tk.LabelFrame(metrics_frame, text="Detailed Analysis", 
                                             pady=10, padx=10)
        detailed_metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.metrics_tree = ttk.Treeview(detailed_metrics_frame, 
                                       columns=("Metric", "Value"), show="headings", height=5)
        self.metrics_tree.heading("Metric", text="Metric")
        self.metrics_tree.heading("Value", text="Value")
        self.metrics_tree.column("Metric", width=150)
        self.metrics_tree.column("Value", width=150)
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)
        
        self.metrics_tree.insert("", "end", values=("Class", "--"))
        self.metrics_tree.insert("", "end", values=("Confidence Score", "--"))
        self.metrics_tree.insert("", "end", values=("Non-Cancerous Probability", "--"))
        self.metrics_tree.insert("", "end", values=("Cancerous Probability", "--"))
        self.metrics_tree.insert("", "end", values=("Decision Threshold", f"{self.config['threshold']}"))
    
    def create_history_tab(self):
        history_frame = tk.Frame(self.history_tab, padx=15, pady=15)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(history_frame, text="Analysis History", 
                font=("Helvetica", 14, "bold"), pady=10).pack()
        ttk.Separator(history_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        columns = ("Date", "Filename", "Result", "Confidence")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=10)
        
        self.history_tree.heading("Date", text="Date")
        self.history_tree.heading("Filename", text="Filename")
        self.history_tree.heading("Result", text="Result")
        self.history_tree.heading("Confidence", text="Confidence")
        
        self.history_tree.column("Date", width=100)
        self.history_tree.column("Filename", width=100)
        self.history_tree.column("Result", width=100)
        self.history_tree.column("Confidence", width=100)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, 
                                command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = tk.Frame(history_frame, pady=10)
        button_frame.pack(fill=tk.X)
        
        self.clear_history_button = ttk.Button(button_frame, text="Clear History", 
                                             command=self.clear_history)
        self.clear_history_button.pack(side=tk.RIGHT)
        
        self.export_history_button = ttk.Button(button_frame, text="Export History", 
                                              command=self.export_history)
        self.export_history_button.pack(side=tk.RIGHT, padx=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                img = Image.open(file_path)
                min_size = min(self.config["target_size"])
                if img.size[0] < min_size or img.size[1] < min_size:
                    messagebox.showwarning("Warning", f"Image must be at least {min_size}x{min_size} pixels.")
                    return
                self.image_path = file_path
                self.load_and_display_image(file_path)
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                self.predict_button.config(state="normal")
                self.image_info_label.config(text=f"Image: {os.path.basename(file_path)}")
                logging.info(f"Image uploaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid image file: {str(e)}")
                logging.error(f"Image upload failed: {str(e)}")
    
    def load_and_display_image(self, file_path):
        self.placeholder_label.place_forget()
        img = Image.open(file_path)
        img = self.resize_image_aspect_ratio(img, (350, 350))
        photo = ImageTk.PhotoImage(img)
        
        if hasattr(self, 'image_label'):
            self.image_label.config(image=photo)
            self.image_label.image = photo
        else:
            self.image_label = tk.Label(self.image_frame, image=photo, bg="white")
            self.image_label.image = photo
            self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def resize_image_aspect_ratio(self, img, max_size):
        width, height = img.size
        ratio = min(max_size[0] / width, max_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def preprocess_image(self, image):
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        input_shape = tuple(self.config["target_size"])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(input_shape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    
    def predict_cancer(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded. Cannot make prediction.")
            return
        if not self.image_path:
            messagebox.showerror("Error", "No image selected.")
            return
        
        # Show progress indicator
        progress = ttk.Progressbar(self.root, mode='indeterminate')
        progress.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        progress.start()
        self.status_var.set("Processing...")
        self.root.update()
        
        try:
            logging.info(f"Predicting for image: {self.image_path}")
            img_pil = Image.open(self.image_path)
            processed_img = self.preprocess_image(img_pil)
            predictions = self.model.predict(processed_img)
            class_index = np.argmax(predictions, axis=1)[0]
            self.confidence_scores = predictions[0]
            
            self.update_results(class_index, self.confidence_scores)
            self.update_metrics(class_index, self.confidence_scores)
            self.add_to_history(class_index, self.confidence_scores)
            
            self.prediction_made = True
            self.status_var.set("Prediction complete")
            self.tab_control.select(0)
            logging.info(f"Prediction completed: Class {class_index}, Confidence {self.confidence_scores}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            logging.error(f"Prediction failed: {str(e)}")
        
        finally:
            progress.stop()
            progress.place_forget()
    
    def update_results(self, class_index, confidence_scores):
        threshold = self.config["threshold"]
        confidence = confidence_scores[class_index]
        if class_index == 1 and confidence >= threshold:
            result_text = "🔴 CANCER DETECTED"
            self.result_label.config(text=result_text, fg="red")
        else:
            result_text = "🟢 NO CANCER DETECTED"
            self.result_label.config(text=result_text, fg="green")
        
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
        self.confidence_bar["value"] = confidence * 100
    
    def update_metrics(self, class_index, confidence_scores):
        self.ax1.clear()
        classes = ["Non-Cancerous", "Cancerous"]
        colors = ["green", "red"]
        bars = self.ax1.barh(classes, confidence_scores, color=colors)
        
        for bar, score in zip(bars, confidence_scores):
            self.ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{score:.2%}", va='center')
        
        self.ax1.set_title("Confidence Scores")
        self.ax1.set_xlim(0, 1.1)
        self.ax1.set_xlabel("Confidence Score")
        self.fig1.tight_layout()
        self.canvas1.draw()
        
        for i, item in enumerate(self.metrics_tree.get_children()):
            if i == 0:
                value = "Cancerous" if class_index == 1 else "Non-Cancerous"
                self.metrics_tree.item(item, values=("Class", value))
            elif i == 1:
                value = f"{confidence_scores[class_index]:.4f}"
                self.metrics_tree.item(item, values=("Confidence Score", value))
            elif i == 2:
                value = f"{confidence_scores[0]:.4f}"
                self.metrics_tree.item(item, values=("Non-Cancerous Probability", value))
            elif i == 3:
                value = f"{confidence_scores[1]:.4f}"
                self.metrics_tree.item(item, values=("Cancerous Probability", value))
    
    def add_to_history(self, class_index, confidence_scores):
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        filename = os.path.basename(self.image_path)
        result = "Cancer" if class_index == 1 else "No Cancer"
        confidence = f"{confidence_scores[class_index]:.2%}"
        self.history_tree.insert("", 0, values=(current_date, filename, result, confidence))
    
    def clear_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        logging.info("History cleared")
    
    def export_history(self):
        if not self.history_tree.get_children():
            messagebox.showinfo("Info", "No history to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export History"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Date,Filename,Result,Confidence\n")
                    for item in self.history_tree.get_children():
                        values = self.history_tree.item(item, "values")
                        f.write(f"{values[0]},{values[1]},{values[2]},{values[3]}\n")
                messagebox.showinfo("Success", f"History exported to {file_path}")
                logging.info(f"History exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export history: {str(e)}")
                logging.error(f"History export failed: {str(e)}")

def main():
    root = tk.Tk()
    app = CancerDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
