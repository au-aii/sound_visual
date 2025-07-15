import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


# MatplotlibがTkinterのウィンドウで動作するようにバックエンドを設定
# この行は、環境によっては不要な場合もあります
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass


def get_word_from_path(file_path):
    """ファイルパスから拡張子を除いたファイル名を取得します。"""
    return os.path.splitext(os.path.basename(file_path))[0]


def analyze_pitch(file_path):
    """
    音声ファイルを分析し、ピッチ曲線グラフを表示します。
    """
    try:
        # 1. 音声ファイルを読み込む
        y, sr = librosa.load(file_path, sr=None)

        # 2. ピッチを抽出する
        # ★★★ エラー修正：最初の引数に 'y' を追加 ★★★
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'), # fminも指定するのが一般的です
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        times = librosa.times_like(f0, sr=sr)

        # 3. グラフを描画する
        plt.figure(figsize=(12, 4))
        plt.plot(times, f0, linewidth=1.5, marker='o', markersize=2) # マーカーも追加して見やすく
        plt.title(f" {get_word_from_path(file_path)}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)
        plt.tight_layout()
        plt.show() # 新しいウィンドウでグラフを表示

    except Exception as e:
        # エラーが発生した場合はメッセージボックスで通知
        messagebox.showerror("エラー", f"ファイルの読み込みまたは解析に失敗しました。\n\n{str(e)}")


def select_file_and_analyze():
    """
    ファイル選択ダイアログを開き、選択されたファイルを分析します。
    """
    # ファイルダイアログはメインのTkウィンドウとは独立して動作可能
    # 先にTkインスタンスを作成しておく
    root = tk.Tk()
    root.withdraw()  # メインウィンドウは非表示にする

    file_path = filedialog.askopenfilename(
        title="解析したい音声ファイルを選択",
        filetypes=[("Audio Files", "*.mp3 *.wav")]
    )
    if file_path:
        analyze_pitch(file_path)
    
    # ダイアログが閉じた後、Tkインスタンスを破棄
    root.destroy()


if __name__ == "__main__":
    # プログラムが実行されたら、ファイル選択関数を呼び出す
    select_file_and_analyze()
