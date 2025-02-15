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
import xlsxwriter

# Set environment variable to allow OpenMP runtime duplication
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder and Analyzer")
        self.root.geometry(f"600x{round((2/3)*self.root.winfo_screenheight())}")  # Set height to match screen height

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

        self.stop_playback_button = tk.Button(control_frame, text="Stop", command=self.stop_audio, width=10)
        self.stop_playback_button.pack(side=tk.LEFT, padx=5)

        self.playback_bar = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, length=400, resolution=0.01, state=tk.DISABLED)
        self.playback_bar.pack(pady=10)

        self.audio_graph_frame = tk.Frame(self.root, height=200)
        self.audio_graph_frame.pack(pady=10, fill=tk.BOTH, expand=False)

        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(pady=0)

        self.equalize_button = tk.Button(self.root, text="Equalize Audio", command=self.equalize_audio, width=20)
        self.equalize_button.pack(pady=10)

        self.hpss_button = tk.Button(self.root, text="Apply HPSS", command=self.apply_hpss, width=20)
        self.hpss_button.pack(pady=10)

        self.fft_button = tk.Button(self.root, text="FFT Analysis", command=self.fft_analysis, width=20)
        self.fft_button.pack(pady=10)

        self.spectrogram_button = tk.Button(self.root, text="Spectrogram", command=self.spectrogram_analysis, width=20)
        self.spectrogram_button.pack(pady=10)


    def start_recording(self):
        if not self.recording:
            self.process_conditions = self.process_entry.get()
            if not self.process_conditions:
                messagebox.showwarning("Warning", "Please enter process conditions before recording.")
                return

            self.recording = True
            self.audio_data = []
            self.start_time = time.time()  # 시작 시간 기록
            self.update_recording_timer()  # 타이머 업데이트 함수 호출

            self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
            self.stream.start()

    def update_recording_timer(self):
        if self.recording:
            elapsed_time = time.time() - self.start_time
            self.record_button.config(text=f"Recording... {elapsed_time:.1f} sec")
            self.root.after(100, self.update_recording_timer)  # 0.1초마다 업데이트
        else:
            self.record_button.config(text="Start Recording")

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.audio_data = np.concatenate(self.audio_data, axis=0)
            self.record_button.config(text="Start Recording")  # 버튼 텍스트 초기화

            # Ask the user where to save the recording
            save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")], title="Save Recording")
            if save_path:
                sf.write(save_path, self.audio_data, self.samplerate)

                # Extract filename from save_path
                filename = os.path.basename(save_path)

                # Append metadata to the CSV file
                metadata = pd.DataFrame([[filename, self.process_conditions]], columns=["filename", "process_conditions"])
                metadata.to_csv(self.metadata_file, mode='a', header=False, index=False)

                messagebox.showinfo("Save", f"Recording saved as {filename} and process conditions added to metadata.")

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.audio_data, self.samplerate = librosa.load(file_path, sr=None, mono=True)
            self.loaded_file_path = file_path
            duration = len(self.audio_data) / self.samplerate
            self.playback_label.config(text=f"Loaded File: {file_path.split('/')[-1]}")
            self.playback_bar.config(state=tk.NORMAL, from_=0, to=duration, resolution=0.01, label=f"0 / {duration:.2f} sec")
            # messagebox.showinfo("Load", "Audio file loaded successfully.")
            
            self.display_audio_waveform()

    def display_audio_waveform(self):
        for widget in self.audio_graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 2), constrained_layout=True)  # Ensure axis labels are visible  # Increase height for better visibility of labels  # Reduce height for fixed placement
        librosa.display.waveshow(self.audio_data, sr=self.samplerate, ax=ax)
        ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

        canvas = FigureCanvasTkAgg(fig, master=self.audio_graph_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()
        canvas.draw()

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

    def stop_audio(self):
        """오디오 재생을 완전히 정지하고 0초부터 다시 시작하도록 설정"""
        if self.playing or self.paused:
            self.playing = False
            self.paused = False
            self.playback_position = 0  # 재생 위치를 처음으로 초기화
            sd.stop()
            self.playback_bar.set(0)
            self.playback_bar.config(label=f"0 / {len(self.audio_data) / self.samplerate:.2f} sec")


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

    def apply_equalization(self, audio, sr):
        """
        특정 주파수 대역의 gain을 적용하여 equalization 수행
        """
        gains = {
            (0, 1000): 0.5,   # 저주파 노이즈 감소
            (1000, 5000): 1.5, # 중주파수 대역 강조
            (5000, 20000): 2.0 # 고주파수 대역 강조
        }
        
        output = np.zeros_like(audio)
        
        for (low, high), gain in gains.items():
            low = max(low, 1e-6)  # 0Hz 방지
            sos = signal.butter(2, [low / (sr / 2), high / (sr / 2)], btype='bandpass', output='sos')
            filtered = signal.sosfilt(sos, audio) * gain
            output += filtered
        
        return output

    def equalize_audio(self):
        """Equalization 후 원본과 비교하는 그래프를 생성하고, -eq 파일로 저장"""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio data available for Equalization.")
            return

        # Equalization 적용
        equalized_audio = self.apply_equalization(self.audio_data, self.samplerate)

        # **파일 자동 저장 (-eq 추가)**
        if self.loaded_file_path:
            file_dir, file_name = os.path.split(self.loaded_file_path)
            file_base, file_ext = os.path.splitext(file_name)
            eq_file_name = f"{file_base}-eq{file_ext}"
            eq_file_path = os.path.join(file_dir, eq_file_name)

            sf.write(eq_file_path, equalized_audio, self.samplerate)  # Equalized 파일 저장
            messagebox.showinfo("Saved", f"Equalized file saved as: {eq_file_path}")

        # **Equalization 후 NaN이나 0값 방지**
        if np.all(equalized_audio == 0):
            messagebox.showerror("Error", "Equalization resulted in all zero values!")
            return
        equalized_audio = np.nan_to_num(equalized_audio)

        # **새로운 Tkinter 창 생성**
        new_window = tk.Toplevel(self.root)
        new_window.title("Equalization Analysis")
        new_window.geometry("1200x800")

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # 3행 2열 서브플롯
        time_axis = np.linspace(0, len(self.audio_data) / self.samplerate, len(self.audio_data))

        # (1) Waveform 비교
        axs[0, 0].plot(time_axis, self.audio_data, label="Raw Audio", alpha=0.7)
        axs[0, 1].plot(time_axis, equalized_audio, label="Equalized Audio", alpha=0.7)
        axs[0, 0].set_title("Waveform (Raw)")
        axs[0, 1].set_title("Waveform (Equalized)")

        max_amplitude = max(np.max(np.abs(self.audio_data)), np.max(np.abs(equalized_audio)))
        axs[0, 0].set_ylim(-max_amplitude, max_amplitude)
        axs[0, 1].set_ylim(-max_amplitude, max_amplitude)

        # (2) FFT 비교
        fft_raw = np.fft.fft(self.audio_data)
        fft_eq = np.fft.fft(equalized_audio)
        freqs = np.fft.fftfreq(len(self.audio_data), 1 / self.samplerate)

        axs[1, 0].plot(freqs[:len(freqs)//2], np.abs(fft_raw[:len(freqs)//2]), label="Raw FFT", alpha=0.7)
        axs[1, 1].plot(freqs[:len(freqs)//2], np.abs(fft_eq[:len(freqs)//2]), label="Equalized FFT", alpha=0.7)
        axs[1, 0].set_xscale("log")
        axs[1, 1].set_xscale("log")

        max_fft_magnitude = max(np.max(np.abs(fft_raw)), np.max(np.abs(fft_eq)))
        axs[1, 0].set_ylim(0, max_fft_magnitude)
        axs[1, 1].set_ylim(0, max_fft_magnitude)

        # (3) Spectrogram 비교 (vmin, vmax 자동 조정)
        _, _, _, im1 = axs[2, 0].specgram(self.audio_data, Fs=self.samplerate, NFFT=1024, cmap="magma")
        _, _, _, im2 = axs[2, 1].specgram(equalized_audio, Fs=self.samplerate, NFFT=1024, cmap="cool")

        vmin = min(im1.get_clim()[0], im2.get_clim()[0])  # 최소값
        vmax = max(im1.get_clim()[1], im2.get_clim()[1])  # 최대값

        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)

        axs[2, 0].set_title("Spectrogram (Raw)")
        axs[2, 1].set_title("Spectrogram (Equalized)")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def apply_hpss(self):
        """Equalized 오디오에 HPSS 적용 후 새로운 파일로 저장하고 비교 그래프를 생성"""
        if self.audio_data is None or self.loaded_file_path is None:
            messagebox.showwarning("Warning", "No audio data available for HPSS.")
            return

        # **Equalized 파일 불러오기**
        file_dir, file_name = os.path.split(self.loaded_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        eq_file_name = f"{file_base}-eq{file_ext}"
        eq_file_path = os.path.join(file_dir, eq_file_name)

        if not os.path.exists(eq_file_path):
            messagebox.showerror("Error", f"Equalized file not found: {eq_file_path}")
            return

        eq_audio, eq_samplerate = librosa.load(eq_file_path, sr=None, mono=True)

        # **HPSS 적용**
        harmonic, percussive = librosa.effects.hpss(eq_audio)
        hpss_audio = harmonic  # 하모닉 성분만 사용

        # **HPSS 파일 저장 (-eq-hpss 추가)**
        hpss_file_name = f"{file_base}-eq-hpss{file_ext}"
        hpss_file_path = os.path.join(file_dir, hpss_file_name)

        sf.write(hpss_file_path, hpss_audio, eq_samplerate)  # HPSS 적용된 파일 저장
        messagebox.showinfo("Saved", f"HPSS-applied file saved as: {hpss_file_path}")

        # **새로운 Tkinter 창 생성**
        new_window = tk.Toplevel(self.root)
        new_window.title("HPSS Analysis")
        new_window.geometry("1200x800")

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # 3행 2열 서브플롯
        time_axis = np.linspace(0, len(eq_audio) / eq_samplerate, len(eq_audio))

        # (1) Waveform 비교
        axs[0, 0].plot(time_axis, eq_audio, label="Equalized Audio", alpha=0.7)
        axs[0, 1].plot(time_axis, hpss_audio, label="Equalized & HPSS Audio", alpha=0.7)
        axs[0, 0].set_title("Waveform (Equalized)")
        axs[0, 1].set_title("Waveform (Equalized & HPSS)")

        max_amplitude = max(np.max(np.abs(eq_audio)), np.max(np.abs(hpss_audio)))
        axs[0, 0].set_ylim(-max_amplitude, max_amplitude)
        axs[0, 1].set_ylim(-max_amplitude, max_amplitude)

        # (2) FFT 비교
        fft_eq = np.fft.fft(eq_audio)
        fft_hpss = np.fft.fft(hpss_audio)
        freqs = np.fft.fftfreq(len(eq_audio), 1 / eq_samplerate)

        axs[1, 0].plot(freqs[:len(freqs)//2], np.abs(fft_eq[:len(freqs)//2]), label="Equalized FFT", alpha=0.7)
        axs[1, 1].plot(freqs[:len(freqs)//2], np.abs(fft_hpss[:len(freqs)//2]), label="Equalized & HPSS FFT", alpha=0.7)
        axs[1, 0].set_xscale("log")
        axs[1, 1].set_xscale("log")

        max_fft_magnitude = max(np.max(np.abs(fft_eq)), np.max(np.abs(fft_hpss)))
        axs[1, 0].set_ylim(0, max_fft_magnitude)
        axs[1, 1].set_ylim(0, max_fft_magnitude)

        # (3) Spectrogram 비교 (vmin, vmax 자동 조정)
        _, _, _, im1 = axs[2, 0].specgram(eq_audio, Fs=eq_samplerate, NFFT=1024, cmap="magma")
        _, _, _, im2 = axs[2, 1].specgram(hpss_audio, Fs=eq_samplerate, NFFT=1024, cmap="cool")

        vmin = min(im1.get_clim()[0], im2.get_clim()[0])  # 최소값
        vmax = max(im1.get_clim()[1], im2.get_clim()[1])  # 최대값

        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)

        axs[2, 0].set_title("Spectrogram (Equalized)")
        axs[2, 1].set_title("Spectrogram (Equalized & HPSS)")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def fft_analysis(self):
        """HPSS까지 처리된 오디오 데이터를 기반으로 FFT 분석 수행 및 저장 기능 추가"""
        
        if self.loaded_file_path is None:
            messagebox.showwarning("Warning", "No audio file loaded for FFT analysis.")
            return

        # **HPSS 파일 불러오기 (-eq-hpss.wav)**
        file_dir, file_name = os.path.split(self.loaded_file_path)
        file_base, file_ext = os.path.splitext(file_name)
        hpss_file_name = f"{file_base}-eq-hpss{file_ext}"
        hpss_file_path = os.path.join(file_dir, hpss_file_name)

        if not os.path.exists(hpss_file_path):
            messagebox.showerror("Error", f"HPSS-processed file not found: {hpss_file_path}")
            return

        # HPSS 적용된 오디오 로드
        audio_data, samplerate = librosa.load(hpss_file_path, sr=None, mono=True)

        # FFT 계산
        N = len(audio_data)
        T = 1.0 / samplerate
        yf = np.fft.fft(audio_data)
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
        new_window.title("FFT Analysis (HPSS Processed)")
        new_window.geometry("800x500")

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf, magnitude, label="FFT Spectrum")
        ax.set_xscale("log")  # 로그 스케일 적용
        ax.set(title="FFT Analysis (HPSS Processed)", xlabel="Frequency (Hz)", ylabel="Amplitude")

        # 상위 주파수 강조 표시
        for i in range(len(top_freqs)):
            ax.axvline(x=top_freqs[i], color='r', linestyle='--', label=f'Peak {i+1}: {top_freqs[i]:.1f} Hz')

        ax.legend()

        # Tkinter에서 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack()

        # 저장 버튼 추가
        button_frame = tk.Frame(new_window)
        button_frame.pack(pady=10)

        # **CSV 저장 함수**
        def save_fft_data():
            save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")],
                                                    title="Save FFT Data")
            if save_path:
                fft_data = pd.DataFrame({"Frequency (Hz)": xf, "Magnitude": magnitude})
                fft_data.to_csv(save_path, index=False)
                messagebox.showinfo("Save", f"FFT Data saved as {save_path}")

        # **PNG 저장 함수**
        def save_fft_graph():
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png")],
                                                    title="Save FFT Graph")
            if save_path:
                fig.savefig(save_path)
                messagebox.showinfo("Save", f"FFT Graph saved as {save_path}")

        # CSV 저장 버튼 추가
        save_data_button = tk.Button(button_frame, text="Save Data to CSV", command=save_fft_data, width=20)
        save_data_button.pack(side=tk.LEFT, padx=10)

        # PNG 저장 버튼 추가
        save_graph_button = tk.Button(button_frame, text="Save Graph as PNG", command=save_fft_graph, width=20)
        save_graph_button.pack(side=tk.RIGHT, padx=10)

        canvas.draw()

        # 결과 출력
        print("🔹 HPSS 처리된 오디오 기준 주요 주파수 목록:")
        for i in range(len(top_freqs)):
            print(f"{i+1}️⃣  {top_freqs[i]:.2f} Hz, 진폭: {top_amplitudes[i]:.6f}")

        return top_freqs, top_amplitudes  # 주요 주파수 및 진폭 반환



    def spectrogram_analysis(self):
        """Raw, Equalized, HPSS된 오디오 데이터를 기반으로 Spectrogram 분석 및 저장 기능 추가"""

        if self.loaded_file_path is None:
            messagebox.showwarning("Warning", "No audio file loaded for Spectrogram analysis.")
            return

        # **파일 경로 설정**
        file_dir, file_name = os.path.split(self.loaded_file_path)
        file_base, file_ext = os.path.splitext(file_name)

        # 원본 (Raw) 오디오 불러오기
        raw_audio, samplerate = librosa.load(self.loaded_file_path, sr=None, mono=True)

        # Equalized 오디오 불러오기 (-eq.wav)
        eq_file_name = f"{file_base}-eq{file_ext}"
        eq_file_path = os.path.join(file_dir, eq_file_name)

        if os.path.exists(eq_file_path):
            eq_audio, _ = librosa.load(eq_file_path, sr=None, mono=True)
        else:
            eq_audio = None  # Equalized 파일이 없을 경우 예외 처리

        # HPSS 적용된 오디오 불러오기 (-eq-hpss.wav)
        hpss_file_name = f"{file_base}-eq-hpss{file_ext}"
        hpss_file_path = os.path.join(file_dir, hpss_file_name)

        if os.path.exists(hpss_file_path):
            hpss_audio, _ = librosa.load(hpss_file_path, sr=None, mono=True)
        else:
            hpss_audio = None  # HPSS 파일이 없을 경우 예외 처리

        # **새로운 Tkinter 창 생성**
        new_window = tk.Toplevel(self.root)
        new_window.title("Spectrogram Analysis (Raw vs. Equalized vs. HPSS)")
        new_window.geometry("1200x900")  # 창 크기 키우기

        # **버튼을 추가할 프레임 생성**
        button_frame = tk.Frame(new_window)
        button_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

        # **Spectrogram 그래프 생성**
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))  # 3개의 Spectrogram 비교

        # **(1) Raw Audio Spectrogram**
        D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(raw_audio)), ref=np.max)
        img1 = librosa.display.specshow(D_raw, sr=samplerate, x_axis='time', y_axis='log', ax=axs[0])
        axs[0].set(title="Spectrogram (Raw Audio)")
        fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

        # **(2) Equalized Audio Spectrogram**
        if eq_audio is not None:
            D_eq = librosa.amplitude_to_db(np.abs(librosa.stft(eq_audio)), ref=np.max)
            img2 = librosa.display.specshow(D_eq, sr=samplerate, x_axis='time', y_axis='log', ax=axs[1])
            axs[1].set(title="Spectrogram (Equalized Audio)")
            fig.colorbar(img2, ax=axs[1], format="%+2.0f dB")
        else:
            axs[1].text(0.5, 0.5, "Equalized Audio Not Found", fontsize=12, ha="center", va="center")

        # **(3) HPSS Audio Spectrogram**
        if hpss_audio is not None:
            D_hpss = librosa.amplitude_to_db(np.abs(librosa.stft(hpss_audio)), ref=np.max)
            img3 = librosa.display.specshow(D_hpss, sr=samplerate, x_axis='time', y_axis='log', ax=axs[2])
            axs[2].set(title="Spectrogram (HPSS Processed Audio)")
            fig.colorbar(img3, ax=axs[2], format="%+2.0f dB")
        else:
            axs[2].text(0.5, 0.5, "HPSS Processed Audio Not Found", fontsize=12, ha="center", va="center")

        plt.tight_layout()

        # **Tkinter Canvas 추가 (그래프 띄우기)**
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # 버튼과 겹치지 않도록 확장

        # **PNG 저장 함수**
        def save_spectrogram_graph():
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png")],
                                                    title="Save Spectrogram Graph")
            if save_path:
                fig.savefig(save_path)
                messagebox.showinfo("Save", f"Spectrogram Graph saved as {save_path}")

        # **CSV 저장 함수**
        def save_spectrogram_data():
            save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    filetypes=[("CSV files", "*.csv")],
                                                    title="Save Spectrogram Data")
            if save_path:
                # 시간-주파수-진폭 데이터 저장
                times = librosa.times_like(D_raw, sr=samplerate)
                freqs = librosa.fft_frequencies(sr=samplerate)

                # 데이터 프레임 생성
                spectrogram_data = pd.DataFrame(D_raw, index=freqs, columns=times)
                spectrogram_data.to_csv(save_path, index=True)
                messagebox.showinfo("Save", f"Spectrogram Data saved as {save_path}")

        # **버튼 추가**
        save_graph_button = tk.Button(button_frame, text="Save Graph as PNG", command=save_spectrogram_graph, width=25)
        save_graph_button.pack(side=tk.LEFT, padx=10)

        save_data_button = tk.Button(button_frame, text="Save Data to CSV", command=save_spectrogram_data, width=25)
        save_data_button.pack(side=tk.RIGHT, padx=10)

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

    def save_data_to_excel(self, data, filename="graph_data.xlsx"):
        """
        주어진 데이터를 pandas를 사용하여 Excel 파일로 저장 (openpyxl 없이)
        """
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel files", "*.xlsx")],
                                                title="Save Data as Excel")
        if save_path:
            df = pd.DataFrame(data)
            df.to_excel(save_path, index=False, engine="xlsxwriter")  # xlsxwriter 엔진 사용
            messagebox.showinfo("Save", f"Data saved as {save_path}")


    def save_graph_as_image(self, fig):
        """
        그래프를 이미지 파일로 저장
        """
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
                                                title="Save Graph as Image")
        if save_path:
            fig.savefig(save_path)
            messagebox.showinfo("Save", f"Graph saved as {save_path}")

    def add_save_buttons(self, parent_window, fig, data):
        """
        그래프 창에 데이터 저장 및 그래프 저장 버튼 추가
        """
        button_frame = tk.Frame(parent_window)
        button_frame.pack(pady=10)

        # 데이터 저장 버튼
        save_data_button = tk.Button(button_frame, text="Save Data to Excel",
                                    command=lambda: self.save_data_to_excel(data),
                                    width=25)
        save_data_button.pack(pady=5)

        # 그래프 저장 버튼
        save_graph_button = tk.Button(button_frame, text="Save Graph as Image",
                                    command=lambda: self.save_graph_as_image(fig),
                                    width=25)
        save_graph_button.pack(pady=5)


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

