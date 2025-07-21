# main3_mfa.py
import os
import librosa
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pyttsx3

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mfa_analyzer import MFA_Analyzer

class PitchAnalyzerMFAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Analyzer with MFA Segmentation")
        self.root.geometry("1000x700")

        self.file_path = None
        self.word = ""
        self.analysis_results = {}
        
        # MFAアナライザーを初期化
        try:
            self.mfa_analyzer = MFA_Analyzer()
        except Exception as e:
            messagebox.showerror("MFA初期化エラー", str(e))
            self.root.destroy()
            return
             
        self.create_widgets()

    def create_widgets(self):
        # 上部コントロールフレーム
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        self.load_button = ttk.Button(control_frame, text="音声ファイルを開く", command=self.select_file)
        self.load_button.pack(side=tk.LEFT, padx=(5))

        self.play_button = ttk.Button(control_frame, text="再生", command=self.play_tts, state="disabled")
        self.play_button.pack(side=tk.LEFT, padx=(5))

        self.file_label = ttk.Label(control_frame, text="ファイルが選択されていません")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # matplotlibグラフエリア
        self.fig = Figure(figsize=(12, 6), dpi=100)
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
                # 分析実行
                self.analysis_results = self._analyze_data_with_mfa(self.file_path)
                self._plot_data_with_segments()
                
                self.file_label.config(text=os.path.basename(self.file_path))
                self.play_button.config(state="normal")

            except Exception as e:
                messagebox.showerror("エラー", f"処理に失敗しました。\n\n{str(e)}")
                self.file_label.config(text="失敗しました")
                self.play_button.config(state="disabled")

    def _analyze_data_with_mfa(self, file_path: str) -> dict:
        """
        【MFA分析】音声ファイルをMFAで音素セグメント化し、ピッチ分析も行う
        """
        print(f"--- MFA Analyzing {os.path.basename(file_path)} ---")
        
        # MFAで音素区間を取得
        segments = self.mfa_analyzer.align_audio_file(file_path)
        
        # librosaでピッチ抽出
        y, sr = librosa.load(file_path, sr=None)
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=1024, hop_length=256
        )
        times = librosa.times_like(f0, sr=sr)
        
        self.word = os.path.splitext(os.path.basename(file_path))[0]
        
        return {
            "f0": f0,
            "times": times,
            "word": self.word,
            "segments": segments
        }

    def _plot_data_with_segments(self):
        """
        【描画】MFA音素セグメントと区画線付きでピッチグラフを描画
        """
        self.ax.clear()
        
        f0 = self.analysis_results.get("f0")
        times = self.analysis_results.get("times")
        word = self.analysis_results.get("word")
        segments = self.analysis_results.get("segments", [])

        if f0 is None or times is None:
            return

        # ピッチグラフを描画
        if not np.isnan(f0).all():
            self.ax.plot(times, f0, linewidth=2, color='black', alpha=0.8)
        else:
            self.ax.text(0.5, 0.5, "ピッチが検出できませんでした", 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=16)

        # 単語を上部に表示
        self.ax.text(0.5, 1.15, word, ha='center', va='center', fontsize=36, 
                    transform=self.ax.transAxes, weight='bold')

        # MFA音素セグメント処理
        if segments:
            # 各音素区間に対して区画線と音素ラベルを配置
            for i, seg in enumerate(segments):
                start_time = seg["start"]
                end_time = seg["end"]
                phoneme = seg["phoneme"]
                center_time = (start_time + end_time) / 2

                # 区画線（音素境界）を描画
                if i > 0:  # 最初以外は境界線を引く
                    self.ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 最後の音素の終端線
                if i == len(segments) - 1:
                    self.ax.axvline(x=end_time, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

                # 音素ラベルを中央揃えで配置
                y_position = np.nanmax(f0) * 1.1 if not np.isnan(f0).all() else 200
                self.ax.text(center_time, y_position, phoneme, 
                           ha='center', va='bottom', fontsize=14, color='red', 
                           weight='bold', family='monospace')

            # X軸の範囲を音素区間に合わせる
            start_range = segments[0]["start"] - 0.05
            end_range = segments[-1]["end"] + 0.05
            self.ax.set_xlim(start_range, end_range)

        self.ax.set_xlabel("Time (s)", fontsize=12)
        self.ax.set_ylabel("Frequency (Hz)", fontsize=12)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.fig.subplots_adjust(top=0.8, bottom=0.15)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = PitchAnalyzerMFAApp(root)
    root.mainloop()