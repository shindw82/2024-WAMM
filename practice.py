import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import librosa
import threading
import numpy as np
import os
import time
import pandas as pd
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio Recorder and Analyzer")
        self.geometry("700x700")
        ctk.set_appearance_mode("dark")  # "light", "dark", "system"
        ctk.set_default_color_theme("blue")

        self.recording = False
        self.audio_data = None
        self.samplerate = 44100
        self.loaded_file_path = None
        self.playing = False
        self.paused = False
        self.playback_thread = None
        self.playback_position = 0
        self.process_conditions = ""
        self.metadata_file = "metadata.csv"
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        self.process_label = ctk.CTkLabel(self, text="3D Printing Process Conditions:", font=("Arial", 14))
        self.process_label.pack(pady=10)

        self.process_entry = ctk.CTkEntry(self, width=400)
        self.process_entry.pack(pady=5)
        
        self.record_button = ctk.CTkButton(self, text="Start Recording", command=self.start_recording)
        self.record_button.pack(pady=10)
        
        self.stop_button = ctk.CTkButton(self, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(pady=10)
        
        self.load_button = ctk.CTkButton(self, text="Load Audio File", command=self.load_audio)
        self.load_button.pack(pady=10)
        
        self.playback_label = ctk.CTkLabel(self, text="No file loaded.", font=("Arial", 12))
        self.playback_label.pack(pady=10)
        
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=5)
        
        self.play_button = ctk.CTkButton(self.control_frame, text="Play", command=self.play_audio)
        self.play_button.pack(side="left", padx=5)
        
        self.pause_button = ctk.CTkButton(self.control_frame, text="Pause", command=self.pause_audio)
        self.pause_button.pack(side="left", padx=5)
        
        self.analyze_button = ctk.CTkButton(self, text="Analyze Audio", command=self.analyze_audio)
        self.analyze_button.pack(pady=10)
        
        self.fft_button = ctk.CTkButton(self, text="FFT Analysis", command=self.fft_analysis)
        self.fft_button.pack(pady=10)
        
        self.extract_freq_button = ctk.CTkButton(self, text="Extract Frequency Signal", command=self.extract_frequency_component)
        self.extract_freq_button.pack(pady=10)
        
        self.advanced_analyze_button = ctk.CTkButton(self, text="Spectrogram Analysis", command=self.advanced_analysis)
        self.advanced_analyze_button.pack(pady=10)
    
    def start_recording(self):
        if not self.recording:
            self.process_conditions = self.process_entry.get()
            if not self.process_conditions:
                messagebox.showwarning("Warning", "Please enter process conditions before recording.")
                return
            
            self.recording = True
            self.audio_data = []
            self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
            self.stream.start()
            messagebox.showinfo("Recording", "Recording started.")

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.audio_data = np.concatenate(self.audio_data, axis=0)
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
            if save_path:
                sf.write(save_path, self.audio_data, self.samplerate)
                messagebox.showinfo("Save", f"Recording saved as {save_path}")
    
    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.audio_data, self.samplerate = librosa.load(file_path, sr=None, mono=True)
            self.loaded_file_path = file_path
            self.playback_label.configure(text=f"Loaded File: {file_path.split('/')[-1]}")
            messagebox.showinfo("Load", "Audio file loaded successfully.")
    
    def play_audio(self):
        if self.audio_data is not None:
            if not self.playing:
                self.playing = True
                self.paused = False
                threading.Thread(target=self.playback).start()
            elif self.paused:
                self.paused = False
                self.playing = True
        else:
            messagebox.showwarning("Warning", "No audio file loaded to play.")
    
    def pause_audio(self):
        if self.playing:
            self.paused = True
            self.playing = False
            sd.stop()
    
    def playback(self):
        try:
            sd.play(self.audio_data, self.samplerate)
            time.sleep(len(self.audio_data) / self.samplerate)
        except Exception as e:
            messagebox.showerror("Error", f"Playback error: {e}")
        finally:
            self.playing = False
    
    def on_closing(self):
        self.destroy()
        print("Application closed.")

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
