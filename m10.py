import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

import librosa
import librosa.display
import numpy as np
import pyttsx3
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# NLTKは重量級ライブラリのため、必要になった時点でインポートを試みる
try:
    import nltk
    # CMU辞書のダウンロードを試みる（初回実行時のみ）
    nltk.download('cmudict', quiet=True)
    CMUDICT = nltk.corpus.cmudict.dict()
except ImportError:
    messagebox.showerror("ライブラリ不足", "NLTKライブラリが見つかりません。\n'pip install nltk' を実行してください。")
    CMUDICT = None
except LookupError:
    messagebox.showerror("データエラー", "NLTKの 'cmudict' データが見つかりません。")
    CMUDICT = None


class PitchAnalyzerApp:
    """
    音声ファイルのピッチを分析し、グラフ表示と音声再生を行うTkinterアプリケーション。
    """
    # --- 定数定義 ---
    PITCH_FMIN = librosa.note_to_hz('C2')
    PITCH_FMAX = librosa.note_to_hz('C7')
    FRAME_LENGTH = 1024
    HOP_LENGTH = 256
    AUDIO_FILE_TYPES = [("Audio Files", "*.mp3 *.wav")]
    TTS_RATE_ADJUSTMENT = -50 # 標準の読み上げ速度からどれだけ遅くするか

    def __init__(self, root: tk.Tk):
        """
        アプリケーションの初期化
        Args:
            root (tk.Tk): Tkinterのルートウィンドウ
        """
        self.root = root
        self.root.title("Pitch Analyzer with Playback")
        self.root.geometry("800x600")

        if CMUDICT is None:
            self.root.destroy()
            return

        # --- インスタンス変数のセットアップ ---
        self.file_path: Optional[str] = None
        self.word_to_speak: str = ""

        # --- GUIのセットアップ ---
        self.create_widgets()

    def create_widgets(self) -> None:
        """GUIのウィジェットを作成し、配置する。"""
        # --- 上部のコントロールフレーム ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)

        self.load_button = ttk.Button(control_frame, text="音声ファイルを開く", command=self.select_and_analyze_file)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))

        self.play_button = ttk.Button(control_frame, text="再生", command=self.play_word_in_thread, state="disabled")
        self.play_button.pack(side=tk.LEFT)

        self.file_label = ttk.Label(control_frame, text="ファイルが選択されていません")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # --- Matplotlibグラフ描画用のキャンバス ---
        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)

    def select_and_analyze_file(self) -> None:
        """ファイル選択ダイアログを開き、選択されたファイルを分析・プロットする。"""
        file_path = filedialog.askopenfilename(
            title="解析したい音声ファイルを選択",
            filetypes=self.AUDIO_FILE_TYPES
        )
        if not file_path:
            return

        self.file_path = file_path
        self.word_to_speak = os.path.splitext(os.path.basename(self.file_path))[0]

        try:
            # 1. 音声データからピッチを抽出
            pitch_data = self._extract_pitch_data(self.file_path)

            if pitch_data:
                f0, times = pitch_data
                phonetic_string = self._get_phonetic_transcription(self.word_to_speak)
                # 2. 抽出したデータでグラフを更新
                self._update_plot(f0, times, self.word_to_speak, phonetic_string)
            else:
                # ピッチが検出できなかった場合
                self._update_plot(None, None, self.word_to_speak, "ピッチ検出不可")

            # 3. UIの状態を更新
            self._update_ui_for_analysis(success=True)

        except Exception as e:
            messagebox.showerror("分析エラー", f"ファイルの分析中にエラーが発生しました。\n\n{e}")
            self._update_ui_for_analysis(success=False)

    def _extract_pitch_data(self, file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        音声ファイルをロードし、ピッチと時間軸を抽出する。
        Args:
            file_path (str): 音声ファイルのパス
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: (f0, times)のタプル、または失敗時にNone
        """
        y, sr = librosa.load(file_path, sr=None)
        f0, _, _ = librosa.pyin(
            y,
            fmin=self.PITCH_FMIN,
            fmax=self.PITCH_FMAX,
            sr=sr,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH
        )
        if np.isnan(f0).all():
            return None # 全てのフレームでピッチが検出できなかった

        times = librosa.times_like(f0, sr=sr, hop_length=self.HOP_LENGTH)
        return f0, times

    def _update_plot(self, f0: Optional[np.ndarray], times: Optional[np.ndarray], word: str, phonetic_str: str) -> None:
        """
        Matplotlibのグラフをクリアし、新しいデータで再描画する。
        Args:
            f0 (Optional[np.ndarray]): 周波数データ
            times (Optional[np.ndarray]): 時間データ
            word (str): 表示する単語
            phonetic_str (str): 表示する発音記号
        """
        self.ax.clear()

        # 単語と発音記号をグラフ上部に表示
        self.ax.text(0.5, 1.12, word, ha='center', va='center', fontsize=40, transform=self.ax.transAxes)
        phoneme_words = phonetic_str.split()
        phoneme_x_positions = np.linspace(0.05, 0.95, len(phoneme_words))
        for p_word, x_pos in zip(phoneme_words, phoneme_x_positions):
            self.ax.text(x_pos, 1.02, p_word, ha='center', va='center', fontsize=16, color='red', transform=self.ax.transAxes)

        # ピッチデータをプロット
        if f0 is not None and times is not None:
            self.ax.plot(times, f0, linewidth=2, marker='o', markersize=3, color='black')
        else:
            self.ax.text(0.5, 0.5, "ピッチが検出できませんでした", ha='center', va='center', transform=self.ax.transAxes, fontsize=16)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.grid(True, linestyle=':')
        self.fig.subplots_adjust(top=0.75) # 上部のテキスト表示スペースを確保
        self.canvas.draw()

    def _update_ui_for_analysis(self, success: bool) -> None:
        """分析結果に応じてUIの状態を更新する。"""
        if success:
            self.file_label.config(text=os.path.basename(self.file_path))
            self.play_button.config(state="normal")
        else:
            self.file_label.config(text="分析に失敗しました")
            self.play_button.config(state="disabled")

    def play_word_in_thread(self) -> None:
        """テキスト読み上げを別スレッドで安全に実行する。"""
        if not self.word_to_speak:
            return
        self.play_button.config(state="disabled")
        # デーモンスレッドにすることで、メインウィンドウを閉じるとスレッドも終了する
        threading.Thread(target=self._tts_task, daemon=True).start()

    def _tts_task(self) -> None:
        """【内部用】pyttsx3での読み上げタスク。バックグラウンドスレッドで実行される。"""
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate + self.TTS_RATE_ADJUSTMENT)
            engine.say(self.word_to_speak)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS再生エラー: {e}")
        finally:
            # GUIの更新はメインスレッドに依頼する (スレッドセーフな方法)
            self.root.after(0, lambda: self.play_button.config(state="normal"))

    def _get_phonetic_transcription(self, word: str) -> str:
        """
        単語からCMU辞書を使って発音記号の文字列を取得する。
        Args:
            word (str): 対象の単語
        Returns:
            str: スペースで区切られた発音記号の文字列
        """
        word_lower = word.lower()
        if CMUDICT and word_lower in CMUDICT:
            # [('W', 'AO1', 'R', 'D')] のような形式から ['W', 'AO', 'R', 'D'] を生成
            phonemes = CMUDICT[word_lower][0]
            # 数字（アクセント情報）を除外して結合
            return ' '.join([''.join(filter(str.isalpha, p)) for p in phonemes])
        return "発音記号が見つかりません"


if __name__ == "__main__":
    app_root = tk.Tk()
    app = PitchAnalyzerApp(app_root)
    app_root.mainloop()