from flask import Flask, render_template, request, jsonify, send_from_directory
import librosa
import matplotlib
matplotlib.use('Agg')  # バックエンド用に設定
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import io
import base64
import tempfile
import json
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# アップロードフォルダの作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class WebPitchAnalyzer:
    def __init__(self):
        """初期化"""
        try:
            self.cmudict_dict = nltk.corpus.cmudict.dict()
        except LookupError:
            print("NLTK cmudictデータが見つかりません。setup_nltk.pyを実行してください。")
            self.cmudict_dict = {}
    
    def analyze_audio_file(self, file_path, filename):
        """音声ファイルを分析"""
        try:
            # librosaで音声を読み込み
            y, sr = librosa.load(file_path, sr=None)
            
            # ピッチ抽出
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                sr=sr, frame_length=1024, hop_length=256
            )
            times = librosa.times_like(f0, sr=sr)
            
            # ファイル名から単語を抽出
            word = os.path.splitext(filename)[0]
            phonetic_string = self.get_phonetic_transcription(word)
            
            # 統計情報を計算
            valid_f0 = f0[~np.isnan(f0)]
            stats = {}
            if len(valid_f0) > 0:
                stats = {
                    'mean_pitch': float(np.mean(valid_f0)),
                    'min_pitch': float(np.min(valid_f0)),
                    'max_pitch': float(np.max(valid_f0)),
                    'pitch_range': float(np.max(valid_f0) - np.min(valid_f0))
                }
            
            return {
                "f0": f0.tolist(),  # JSONシリアライズのためリストに変換
                "times": times.tolist(),
                "word": word,
                "phonetic_string": phonetic_string,
                "stats": stats,
                "duration": float(librosa.get_duration(y=y, sr=sr))
            }
            
        except Exception as e:
            print(f"音声分析エラー: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def get_phonetic_transcription(self, word):
        """発音記号を取得"""
        word_lower = word.lower()
        if word_lower in self.cmudict_dict:
            phonemes = self.cmudict_dict[word_lower][0]
            return ' '.join([''.join(filter(str.isalpha, p)) for p in phonemes])
        else:
            return "発音記号が見つかりません"
    
    def create_pitch_plot(self, analysis_result):
        """ピッチグラフを作成してbase64エンコードして返す"""
        try:
            f0 = np.array(analysis_result["f0"])
            times = np.array(analysis_result["times"])
            word = analysis_result["word"]
            phonetic_string = analysis_result["phonetic_string"]
            
            # matplotlib図を作成
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            
            # 有効なピッチデータのみプロット
            valid_mask = ~np.isnan(f0)
            if np.any(valid_mask):
                ax.plot(times[valid_mask], f0[valid_mask], 
                       'b-', linewidth=2, marker='o', markersize=3, alpha=0.8)
                ax.set_ylim(np.min(f0[valid_mask]) * 0.9, np.max(f0[valid_mask]) * 1.1)
            else:
                ax.text(0.5, 0.5, "ピッチが検出できませんでした", 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=16, color='red')
            
            # タイトルと発音記号
            ax.text(0.5, 1.12, word, ha='center', va='center', 
                   fontsize=32, transform=ax.transAxes, weight='bold')
            
            # 発音記号を分割して配置
            phoneme_words = phonetic_string.split()
            if len(phoneme_words) > 0:
                phoneme_x_positions = np.linspace(0.05, 0.95, len(phoneme_words))
                for p_word, x_pos in zip(phoneme_words, phoneme_x_positions):
                    ax.text(x_pos, 1.02, p_word, ha='center', va='center', 
                           fontsize=14, color='red', transform=ax.transAxes,
                           family='monospace')
            
            # 軸設定
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Frequency (Hz)", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
            
            # レイアウト調整
            plt.subplots_adjust(top=0.8, bottom=0.15)
            plt.tight_layout()
            
            # base64エンコード
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                       dpi=150, facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig)  # メモリリークを防ぐ
            
            return img_base64
            
        except Exception as e:
            print(f"グラフ作成エラー: {str(e)}")
            print(traceback.format_exc())
            return None

# グローバルインスタンス
analyzer = WebPitchAnalyzer()

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """ファイルアップロードと分析"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'ファイルが選択されていません'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'ファイルが選択されていません'})
        
        # ファイル形式チェック
        allowed_extensions = {'mp3', 'wav', 'ogg', 'm4a'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': '対応していないファイル形式です'})
        
        # 安全なファイル名を生成
        filename = secure_filename(file.filename)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # 音声分析実行
            analysis_result = analyzer.analyze_audio_file(tmp_path, filename)
            
            if analysis_result is None:
                return jsonify({'success': False, 'error': '音声分析に失敗しました'})
            
            # グラフ作成
            plot_image = analyzer.create_pitch_plot(analysis_result)
            
            if plot_image is None:
                return jsonify({'success': False, 'error': 'グラフ作成に失敗しました'})
            
            # 成功レスポンス
            return jsonify({
                'success': True,
                'word': analysis_result['word'],
                'phonetic': analysis_result['phonetic_string'],
                'plot': plot_image,
                'stats': analysis_result['stats'],
                'duration': analysis_result['duration']
            })
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"アップロード処理エラー: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'処理中にエラーが発生しました: {str(e)}'})

@app.route('/static/<path:filename>')
def static_files(filename):
    """静的ファイルの配信"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # ポートを5000から8080に変更