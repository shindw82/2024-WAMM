import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import torchaudio
import torchaudio.transforms as T
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import os
import sys
import pandas as pd
import scipy.signal as signal

# Set environment variable to allow OpenMP runtime duplication
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder and Analyzer")
        self.root.geometry("600x600")

        self.recording = False
        self.audio_data = None
        self.samplerate = 44100
        self.loaded_file_path = None
        self.playing = False
        self.paused = False
        self.playback_thread = None
        self.playback_position = 0
        self.process_conditions = ""

        self.create_widgets()

        # 창 닫기 이벤트 핸들러 등록
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize metadata file
        self.exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
        self.metadata_file = os.path.join(self.exe_dir, "metadata.csv")
        if not os.path.exists(self.metadata_file):
            pd.DataFrame(columns=["filename", "process_conditions"]).to_csv(self.metadata_file, index=False)

    def create_widgets(self):
        self.process_label = tk.Label(self.root, text="3D Printing Process Conditions:", font=("Arial", 10))
        self.process_label.pack(pady=5)

        self.process_entry = tk.Entry(self.root, width=50)
        self.process_entry.pack(pady=5)

        self.record_button = tk.Button(self.root, text="Start Recording", command=self.start_recording, width=20)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, width=20)
        self.stop_button.pack(pady=10)

        self.load_button = tk.Button(self.root, text="Load Audio File", command=self.load_audio, width=20)
        self.load_button.pack(pady=10)

        self.playback_label = tk.Label(self.root, text="No file loaded.", font=("Arial", 10))
        self.playback_label.pack(pady=10)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        self.play_button = tk.Button(control_frame, text="Play", command=self.play_audio, width=10)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(control_frame, text="Pause", command=self.pause_audio, width=10)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.playback_bar = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=0.01, state=tk.DISABLED)
        self.playback_bar.pack(pady=10)

        self.analyze_button = tk.Button(self.root, text="Analyze Audio", command=self.analyze_audio, width=20)
        self.analyze_button.pack(pady=10)

        self.fft_button = tk.Button(self.root, text="FFT Analysis", command=self.fft_analysis, width=20)
        self.fft_button.pack(pady=10)

        self.extract_freq_button = tk.Button(self.root, text="Extract Frequency Signal", command=self.extract_frequency_component, width=25)
        self.extract_freq_button.pack(pady=10)

        self.advanced_analyze_button = tk.Button(self.root, text="Spectrogram Analysis", command=self.advanced_analysis, width=20)
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

            # Ask the user where to save the recording
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")], title="Save Recording")
            if save_path:
                sf.write(save_path, self.audio_data, self.samplerate)

                # Append metadata to the CSV file
                metadata = pd.DataFrame([[save_path, self.process_conditions]], columns=["filename", "process_conditions"])
                metadata.to_csv(self.metadata_file, mode='a', header=False, index=False)

                messagebox.showinfo("Save", f"Recording saved as {save_path} and process conditions added to metadata.")

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.audio_data, self.samplerate = librosa.load(file_path, sr=None, mono=True)
            self.loaded_file_path = file_path
            duration = len(self.audio_data) / self.samplerate
            self.playback_label.config(text=f"Loaded File: {file_path.split('/')[-1]}")
            self.playback_bar.config(state=tk.NORMAL, from_=0, to=duration, resolution=0.01, label=f"0 / {duration:.2f} sec")
            messagebox.showinfo("Load", "Audio file loaded successfully.")

    def play_audio(self):
        if self.audio_data is not None:
            if not self.playing:
                self.playing = True
                self.paused = False
                self.playback_thread = threading.Thread(target=self.playback)
                self.playback_thread.start()
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
            start_index = int(self.playback_position * self.samplerate)
            duration = len(self.audio_data) / self.samplerate
            sd.play(self.audio_data[start_index:], self.samplerate)
            start_time = time.time() - self.playback_position

            while self.playing and self.playback_position < duration:
                if self.paused:
                    sd.stop()
                    break

                elapsed_time = time.time() - start_time
                self.playback_position = elapsed_time
                self.playback_bar.set(self.playback_position)
                self.playback_bar.config(label=f"{self.playback_position:.2f} / {duration:.2f} sec")
                time.sleep(0.01)

            if not self.paused:
                self.playing = False
                self.playback_position = 0
                self.playback_bar.set(0)
                self.playback_bar.config(label=f"0 / {duration:.2f} sec")

        except Exception as e:
            messagebox.showerror("Error", f"Playback error: {e}")
        finally:
            self.playing = False

    def analyze_audio(self):
        if self.audio_data is not None:
            # 새로운 팝업 창 생성
            new_window = tk.Toplevel(self.root)
            new_window.title("Waveform Analysis")
            new_window.geometry("800x400")

            fig, ax = plt.subplots(figsize=(8, 4))
            librosa.display.waveshow(self.audio_data, sr=self.samplerate, ax=ax)
            ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()
            canvas.draw()
        else:
            messagebox.showwarning("Warning", "No audio data to analyze.")

    def fft_analysis(self):
        """FFT 분석 - 가장 강한 상위 5개의 주파수를 찾고 시각화"""
        
        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio data available for FFT analysis.")
            return
        
        # 샘플레이트를 self.samplerate에서 가져오기
        samplerate = self.samplerate  

        # FFT 계산
        N = len(self.audio_data)
        T = 1.0 / samplerate
        yf = np.fft.fft(self.audio_data)
        xf = np.fft.fftfreq(N, T)[:N//2]  # 양의 주파수 성분만 사용
        magnitude = 2.0/N * np.abs(yf[:N//2])  # 진폭 계산
        
        # 가장 강한 주파수 찾기
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitude, height=0.00001)  # 특정 진폭 이상인 주파수 찾기
        peak_magnitudes = magnitude[peaks]

        # 상위 5개 주파수 찾기
        top_n = 5
        top_indices = np.argsort(peak_magnitudes)[-top_n:][::-1]  # 높은 순으로 정렬
        top_freqs = xf[peaks][top_indices]  # 상위 주파수들
        top_amplitudes = peak_magnitudes[top_indices]  # 상위 진폭들

        # 새로운 창 생성
        new_window = tk.Toplevel(self.root)
        new_window.title("FFT Analysis")
        new_window.geometry("800x400")

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf, magnitude, label="FFT Spectrum")
        ax.set_xscale("log")  # 로그 스케일 적용
        ax.set(title="FFT Analysis (Top Frequencies)", xlabel="Frequency (Hz)", ylabel="Amplitude")

        # 상위 주파수 강조 표시
        for i in range(len(top_freqs)):
            ax.axvline(x=top_freqs[i], color='r', linestyle='--', label=f'Peak {i+1}: {top_freqs[i]:.1f} Hz')

        ax.legend()

        # Tkinter에서 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

        # 결과 출력
        print("🔹 주요 주파수 목록:")
        for i in range(len(top_freqs)):
            print(f"{i+1}️⃣  {top_freqs[i]:.2f} Hz, 진폭: {top_amplitudes[i]:.6f}")

        return top_freqs, top_amplitudes  # 주요 주파수 및 진폭 반환


    def advanced_analysis(self):
        """스펙트로그램 분석을 수행하는 함수"""
        if self.audio_data is not None:
            new_window = tk.Toplevel(self.root)
            new_window.title("Spectrogram Analysis")
            new_window.geometry("800x400")

            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
            img = librosa.display.specshow(D, sr=self.samplerate, x_axis='time', y_axis='log', ax=ax)
            ax.set(title="Spectrogram Analysis")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            canvas = FigureCanvasTkAgg(fig, master=new_window)
            canvas.get_tk_widget().pack()
            canvas.draw()
        else:
            messagebox.showwarning("Warning", "No audio data available for spectrogram analysis.")


    def extract_frequency_component(self):
        """ 특정 주파수 성분만 추출하여 시간-진폭 그래프로 표시하는 함수 """

        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio data available for extraction.")
            return

        # 사용자 입력: 특정 주파수를 입력받음
        target_freq = simpledialog.askfloat("Input", "Enter target frequency (Hz):", minvalue=0.1)
        if target_freq is None:
            return  # 입력이 취소된 경우 함수 종료

        # 샘플레이트 가져오기
        samplerate = self.samplerate

        # FFT 수행
        N = len(self.audio_data)
        freqs = np.fft.fftfreq(N, 1 / samplerate)
        fft_spectrum = np.fft.fft(self.audio_data)

        # 관심 주파수 대역 (±2 Hz)
        bandwidth = 2
        lower_bound = target_freq - bandwidth
        upper_bound = target_freq + bandwidth

        # 특정 주파수만 남기고 나머지 제거
        filtered_spectrum = np.zeros_like(fft_spectrum)
        mask = (freqs >= lower_bound) & (freqs <= upper_bound)
        filtered_spectrum[mask] = fft_spectrum[mask]

        # 역 FFT (IFFT) 수행 → 시간 도메인 신호 변환
        extracted_signal = np.fft.ifft(filtered_spectrum).real

        # 시간 축 생성
        time_axis = np.linspace(0, N / samplerate, N)

        # 새로운 창에서 그래프 표시
        new_window = tk.Toplevel(self.root)
        new_window.title(f"Extracted {target_freq:.2f} Hz Signal")
        new_window.geometry("800x400")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_axis, extracted_signal, label=f"Filtered {target_freq:.2f} Hz Signal", color='r')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Extracted {target_freq:.2f} Hz Signal in Time Domain")
        ax.legend()
        ax.grid()

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()
        canvas.draw()


    def on_closing(self):
        """창이 닫힐 때 실행되는 함수"""
        if self.playing:
            self.playing = False  # 재생 중이면 중지
            sd.stop()  # 사운드 정지
            time.sleep(0.1)  # 약간의 지연으로 안전 종료

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)  # 스레드 종료 대기

        self.root.quit()  # Tkinter 루프 종료
        self.root.destroy()  # 모든 GUI 요소 제거
        print("프로그램이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    print("Initializing the GUI...")

    # 디렉토리 설정: 경로 문제 방지
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
        os.chdir(exe_dir)
        print(f"Current working directory: {exe_dir}")
    except Exception as e:
        print(f"Failed to change directory: {e}")

    try:
        # GUI 초기화
        root = tk.Tk()
        app = AudioApp(root)
        print("Starting the GUI application.")
        root.mainloop()
    except Exception as e:
        print(f"Error occurred during GUI initialization: {e}")

