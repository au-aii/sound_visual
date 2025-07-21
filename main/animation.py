
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nltk
import numpy as np
import threading
import pyttsx3

# MatplotlibのグラフをTkinterに埋め込むための部品をインポート
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

class PitchAnimatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Animator")
        self.root.geometry("800x600")

        # --- アプリケーションの状態を保持する変数 ---
        self.file_path = None
        self.word = ""
        self.f0_data = None
        self.time_data = None
        self.animation = None
        
        # ★アニメーション用の設定値
        self.min_f0, self.max_f0 = 100, 300  # デフォルトのピッチ範囲
        self.min_fontsize, self.max_fontsize = 20, 70 # フォントサイズの範囲

        # --- NLTK辞書のロード ---
        try:
            self.cmudict_dict = nltk.corpus.cmudict.dict()
        except LookupError:
            messagebox.showerror("データエラー", "NLTKの 'cmudict' データが見つかりません。")
            self.root.destroy()
            return
        
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        self.load_button = ttk.Button(control_frame, text="音声ファイルを開く", command=self.select_file_and_analyze)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))

        self.play_button = ttk.Button(control_frame, text="▶️ アニメーション再生", command=self.start_animation_and_audio, state="disabled")
        self.play_button.pack(side=tk.LEFT)

        self.file_label = ttk.Label(control_frame, text="ファイルが選択されていません")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file_and_analyze(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
        if file_path:
            self.file_path = file_path
            self.prepare_plot()

    def prepare_plot(self):
        """静的なグラフとアニメーションの準備を行う"""
        if not self.file_path: return
        try:
            y, sr = librosa.load(self.file_path, sr=None)
            self.f0_data, _, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                sr=sr, frame_length=1024, hop_length=256
            )
            self.time_data = librosa.times_like(self.f0_data, sr=sr)
            self.word = os.path.splitext(os.path.basename(self.file_path))[0]
            
            # ★変更点: ピッチの範囲を計算して保持
            f0_valid = self.f0_data[~np.isnan(self.f0_data)]
            if len(f0_valid) > 1:
                self.min_f0 = f0_valid.min()
                self.max_f0 = f0_valid.max()
            
            self.ax.clear()
            # 静的なピッチ曲線は薄く表示
            self.ax.plot(self.time_data, self.f0_data, linewidth=1.5, color='lightblue', alpha=0.7)
            
            # ★変更点: アニメーション用テキストの初期位置を固定
            # テキストのY座標を、ピッチの平均値あたりに固定する
            y_center = np.nanmean(f0_valid) if len(f0_valid) > 0 else 150
            self.animated_text = self.ax.text(
                self.time_data.mean(), y_center, self.word,
                ha='center', va='center', fontsize=self.min_fontsize, 
                color='black', family='monospace', weight='bold'
            )
            self.progress_line, = self.ax.plot([], [], color='red', lw=2, alpha=0.8)

            # 静的な発音記号の表示
            phonetic_string = self.get_phonetic_transcription(self.word)
            phoneme_words = phonetic_string.split()
            phoneme_x_positions = np.linspace(0.05, 0.95, len(phoneme_words))
            for p_word, x_pos in zip(phoneme_words, phoneme_x_positions):
                self.ax.text(x_pos, 0.05, p_word, ha='center', va='bottom', fontsize=16, color='gray', transform=self.ax.transAxes)

            self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Frequency (Hz)")
            self.ax.grid(True, linestyle=':'); self.fig.tight_layout()
            self.canvas.draw()

            self.file_label.config(text=os.path.basename(self.file_path))
            self.play_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("エラー", f"分析に失敗しました。\n\n{str(e)}")

    def start_animation_and_audio(self):
        """アニメーションと音声再生を同時に開始する"""
        if self.time_data is None or len(self.time_data) == 0: return
        
        self.play_button.config(state="disabled")
        threading.Thread(target=self._play_tts_task, daemon=True).start()

        duration = self.time_data[-1]
        fps = 30 # フレームレートを少し調整
        total_frames = int(duration * fps)

        self.animation = FuncAnimation(
            self.fig, self.update_animation, frames=total_frames,
            interval=1000/fps, blit=True, repeat=False
        )
        self.canvas.draw()

    def update_animation(self, frame_num):
        """アニメーションの各フレームで呼び出される更新関数"""
        fps = 30
        current_time = frame_num / fps
        
        idx = np.searchsorted(self.time_data, current_time)
        if idx >= len(self.f0_data):
            return self.animated_text, self.progress_line

        # ★★★ ここからが新しいアニメーションロジック ★★★
        # 1. テキストのX座標を更新
        x, y = self.animated_text.get_position()
        self.animated_text.set_position((current_time, y))

        # 2. ピッチに応じてフォントサイズを更新
        current_f0 = self.f0_data[idx]
        if not np.isnan(current_f0):
            # ピッチの値をフォントサイズの範囲にマッピング
            pitch_range = self.max_f0 - self.min_f0
            if pitch_range > 0:
                normalized_pitch = (current_f0 - self.min_f0) / pitch_range
                # 非線形マッピングで、変化をよりダイナミックに
                font_size = self.min_fontsize + (normalized_pitch ** 0.5) * (self.max_fontsize - self.min_fontsize)
                self.animated_text.set_fontsize(font_size)
        else:
            # ピッチがない区間は最小サイズにする
            self.animated_text.set_fontsize(self.min_fontsize)
        
        self.progress_line.set_data([current_time, current_time], self.ax.get_ylim())
        return self.animated_text, self.progress_line

    def _play_tts_task(self):
        """【内部用】pyttsx3での読み上げタスク"""
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)
            engine.say(self.word)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS再生エラー: {e}")
        finally:
            self.root.after(100, lambda: self.play_button.config(state="normal"))

    def get_phonetic_transcription(self, word):
        word_lower = word.lower()
        if word_lower in self.cmudict_dict:
            phonemes = self.cmudict_dict[word_lower][0]
            return ' '.join([''.join(filter(str.isalpha, p)) for p in phonemes])
        else:
            return "発音記号が見つかりません"

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAnimatorApp(root)
    root.mainloop()

