import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nltk
import numpy as np
import pyttsx3

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PitchAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Analyzer with Playback")
        self.root.geometry("800x600")

        self.file_path = None
        self.word = ""
        self.analysis_results = {} 

        self.cmudict_dict = nltk.corpus.cmudict.dict()
             
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        self.load_button = ttk.Button(control_frame, text="open", command=self.select_file)
        self.load_button.pack(side=tk.LEFT, padx=(5))

        self.play_button = ttk.Button(control_frame, text="play", command=self.play_tts, state="disabled")
        self.play_button.pack(side=tk.LEFT)

        self.file_label = ttk.Label(control_frame, text="File is not selected")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="解析したい音声ファイルを選択",
            filetypes=[("Audio Files", "*.mp3 *.wav")]
        )
        if file_path:
            self.file_path = file_path
            
            try:
                self.analysis_results = self._analyze_data(self.file_path)
                self._plot_data()
                
                self.file_label.config(text=os.path.basename(self.file_path))
                self.play_button.config(state="normal")

            except Exception as e:
                messagebox.showerror("エラー", f"処理に失敗しました。\n\n{str(e)}")
                self.file_label.config(text="失敗しました")
                self.play_button.config(state="disabled")

    def _analyze_data(self, file_path: str) -> dict:
        """
        【分析】音声ファイルを分析し、結果を辞書として返す。
        """
        print(f"--- Analyzing {os.path.basename(file_path)} ---")
        y, sr = librosa.load(file_path, sr=None)
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=1024, hop_length=256
        )
        times = librosa.times_like(f0, sr=sr)
        
        self.word = os.path.splitext(os.path.basename(file_path))[0]
        phonetic_string = self.get_phonetic_transcription(self.word)
        
        # 分析結果をまとめて返す
        return {
            "f0": f0,
            "times": times,
            "word": self.word,
            "phonetic_string": phonetic_string
        }

    def _plot_data(self):
        self.ax.clear()
        
        # 描画に必要なデータを取得
        f0 = self.analysis_results.get("f0")
        times = self.analysis_results.get("times")
        word = self.analysis_results.get("word")
        phonetic_string = self.analysis_results.get("phonetic_string")

        if f0 is None or times is None:
            return 
        
        if np.isnan(f0).all():
            self.ax.text(0.5, 0.5, "ピッチが検出できませんでした", ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
        else:
            self.ax.plot(times, f0, linewidth=2, marker='o', markersize=3, color='black')

        # テキストを配置
        self.ax.text(0.5, 1.12, word, ha='center', va='center', fontsize=40, transform=self.ax.transAxes)
        phoneme_words = phonetic_string.split()
        phoneme_x_positions = np.linspace(0.05, 0.95, len(phoneme_words))
        for p_word, x_pos in zip(phoneme_words, phoneme_x_positions):
            self.ax.text(x_pos, 1.02, p_word, ha='center', va='center', fontsize=16, color='red', transform=self.ax.transAxes)
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.grid(True, linestyle=':')
        self.fig.subplots_adjust(top=0.75)
        
        self.canvas.draw()

    def play_tts(self):
    
        if not self.word:
            return
            
        self.play_button.config(state="disabled")
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)
            engine.say(self.word)

            engine.runAndWait()
        except Exception as e:
            print(f"TTS再生エラー: {e}")
        finally:
            self.play_button.config(state="normal")
            
    def get_phonetic_transcription(self, word):
        word_lower = word.lower()
        if word_lower in self.cmudict_dict:
            phonemes = self.cmudict_dict[word_lower][0]
            return ' '.join([''.join(filter(str.isalpha, p)) for p in phonemes])
        else:
            return "Phonetic symbol not found"

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAnalyzerApp(root)
    root.mainloop()