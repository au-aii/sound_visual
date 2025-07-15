import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nltk
import numpy as np
import threading
import pyttsx3  # テキスト読み上げライブラリ

# MatplotlibのグラフをTkinterに埋め込むための部品をインポート
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PitchAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Analyzer with Playback")
        self.root.geometry("800x600")

        # --- 状態変数 ---
        self.file_path = None
        self.word = "" # 読み上げる単語

        # --- NLTK load ---
        try:
            self.cmudict_dict = nltk.corpus.cmudict.dict()
            print("NLTK cmudict successfully loaded.")
        except LookupError:
            messagebox.showerror("data error")
            self.root.destroy()
            return
        
        # ---ウィジェットの作成 ---
        self.create_widgets()

    def create_widgets(self):
        # --- heder ---
        heder = ttk.Frame(self.root, padding="10")
        heder.pack(fill=tk.X)

        self.load_button = ttk.Button(heder, text="音声ファイルを開く", command=self.select_file_and_analyze)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))

        # 再生ボタンを一つに統一
        self.play_button = ttk.Button(heder, text="再生", command=self.play_tts, state="disabled")
        self.play_button.pack(side=tk.LEFT)

        self.file_label = ttk.Label(heder, text="ファイルが選択されていません")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # --- グラフ描画 canvas ---
        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file_and_analyze(self):
        file_path = filedialog.askopenfilename(
            title="解析したい音声ファイルを選択",
            filetypes=[("Audio Files", "*.mp3 *.wav")]
        )
        if file_path:
            self.file_path = file_path
            self.analyze_and_plot()

    def analyze_and_plot(self):
        if not self.file_path: return
        try:
            y, sr = librosa.load(self.file_path, sr=None)
            f0, _, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                sr=sr, frame_length=1024, hop_length=256
            )
            times = librosa.times_like(f0, sr=sr)

            self.word = os.path.splitext(os.path.basename(self.file_path))[0]
            phonetic_string = self.get_phonetic_transcription(self.word)
            
            self.ax.clear()
            if np.isnan(f0).all():
                self.ax.text(0.5, 0.5, "ピッチが検出できませんでした", ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
            else:
                self.ax.plot(times, f0, linewidth=2, marker='o', markersize=3, color='black')

            self.ax.text(0.5, 1.12, self.word, ha='center', va='center', fontsize=40, transform=self.ax.transAxes)
            phoneme_words = phonetic_string.split()
            phoneme_x_positions = np.linspace(0.05, 0.95, len(phoneme_words))
            for p_word, x_pos in zip(phoneme_words, phoneme_x_positions):
                self.ax.text(x_pos, 1.02, p_word, ha='center', va='center', fontsize=16, color='red', transform=self.ax.transAxes)
            
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Frequency (Hz)")
            self.ax.grid(True, linestyle=':')
            self.fig.subplots_adjust(top=0.75)
            self.canvas.draw()

            self.file_label.config(text=os.path.basename(self.file_path))
            self.play_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("エラー", f"分析に失敗しました。\n\n{str(e)}")
            self.file_label.config(text="分析に失敗しました")
            self.play_button.config(state="disabled")

    def play_tts(self):
        """テキスト読み上げを別スレッドで実行する"""
        if not self.word: return
        self.play_button.config(state="disabled")
        threading.Thread(target=self._play_tts_task).start()

    def _play_tts_task(self):
        """【内部用】pyttsx3での読み上げタスク"""
        try:
            engine = pyttsx3.init()
            
            # 読上速度調整
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)
            
            engine.say(self.word)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS再生エラー: {e}")
        finally:
            # GUIの更新はメインスレッドに安全に依頼します。
            self.root.after(0, lambda: self.play_button.config(state="normal"))

    def get_phonetic_transcription(self, word):
        word_lower = word.lower()
        if word_lower in self.cmudict_dict:
            phonemes = self.cmudict_dict[word_lower][0]
            return ' '.join([''.join(filter(str.isalpha, p)) for p in phonemes])
        else:
            return "発音記号が見つかりません"

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAnalyzerApp(root)
    root.mainloop()
