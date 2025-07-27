class PitchAnalyzerApp {
    constructor() {
        this.init();
    }

    init() {
        // DOM要素の取得
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.errorMessage = document.getElementById('errorMessage');

        // イベントリスナーの設定
        this.setupEventListeners();
    }

    setupEventListeners() {
        // ファイル選択
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // ドラッグ&ドロップ
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });

        // アップロードエリアクリック
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });
    }

    async handleFileUpload(file) {
        // ファイル形式チェック
        const allowedTypes = ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['mp3', 'wav', 'ogg', 'm4a'];

        if (!allowedExtensions.includes(fileExtension)) {
            this.showError('対応していないファイル形式です。MP3, WAV, OGG, M4A形式のファイルを選択してください。');
            return;
        }

        // ファイルサイズチェック (16MB制限)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('ファイルサイズが大きすぎます。16MB以下のファイルを選択してください。');
            return;
        }

        try {
            // UIを更新
            this.showLoading();

            // FormDataの作成
            const formData = new FormData();
            formData.append('file', file);

            // サーバーへアップロード
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // 成功時の処理
                await this.displayResults(result, file);
            } else {
                // エラー時の処理
                this.showError(result.error || '不明なエラーが発生しました');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showError('ネットワークエラーが発生しました。インターネット接続を確認してください。');
        }
    }

    showLoading() {
        // 全セクションを非表示
        this.hideAllSections();
        // ローディングセクションを表示
        this.loadingSection.style.display = 'block';
    }

    async displayResults(result, file) {
        try {
            // 全セクションを非表示
            this.hideAllSections();

            // 結果データを表示
            document.getElementById('wordText').textContent = result.word;
            document.getElementById('phoneticText').textContent = result.phonetic;

            // 統計情報を表示
            if (result.stats && Object.keys(result.stats).length > 0) {
                document.getElementById('meanPitch').textContent = `${result.stats.mean_pitch.toFixed(1)} Hz`;
                document.getElementById('pitchRange').textContent = 
                    `${result.stats.min_pitch.toFixed(1)} - ${result.stats.max_pitch.toFixed(1)} Hz`;
            } else {
                document.getElementById('meanPitch').textContent = 'N/A';
                document.getElementById('pitchRange').textContent = 'N/A';
            }

            document.getElementById('duration').textContent = `${result.duration.toFixed(2)} 秒`;

            // グラフを表示
            const plotImage = document.getElementById('plotImage');
            plotImage.src = `data:image/png;base64,${result.plot}`;
            plotImage.style.display = 'block';

            // 音声プレーヤーを追加
            this.addAudioPlayer(file);

            // 結果セクションを表示
            this.resultsSection.style.display = 'block';

        } catch (error) {
            console.error('Results display error:', error);
            this.showError('結果の表示中にエラーが発生しました');
        }
    }

    addAudioPlayer(file) {
        const audioPlayer = document.getElementById('audioPlayer');
        
        // 既存のプレーヤーを削除
        audioPlayer.innerHTML = '';

        // 新しい音声プレーヤーを作成
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.style.width = '100%';
        audio.style.maxWidth = '500px';
        
        // ファイルのURLを作成
        const audioURL = URL.createObjectURL(file);
        audio.src = audioURL;

        // プレーヤーを追加
        audioPlayer.appendChild(audio);

        // メモリリークを防ぐため、音声が終了したらURLを解放
        audio.addEventListener('ended', () => {
            URL.revokeObjectURL(audioURL);
        });
    }

    showError(message) {
        // 全セクションを非表示
        this.hideAllSections();
        
        // エラーメッセージを設定
        this.errorMessage.textContent = message;
        
        // エラーセクションを表示
        this.errorSection.style.display = 'block';
    }

    hideAllSections() {
        const sections = [
            this.loadingSection,
            this.resultsSection,
            this.errorSection
        ];
        
        sections.forEach(section => {
            if (section) {
                section.style.display = 'none';
            }
        });
    }
}

// アプリケーションのリセット
function resetApp() {
    // ファイル入力をクリア
    document.getElementById('fileInput').value = '';
    
    // 全セクションを非表示
    const sections = ['loadingSection', 'resultsSection', 'errorSection'];
    sections.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.style.display = 'none';
        }
    });
    
    // 音声プレーヤーをクリア
    const audioPlayer = document.getElementById('audioPlayer');
    if (audioPlayer) {
        audioPlayer.innerHTML = '';
    }
}

// ページ読み込み完了時にアプリケーションを初期化
document.addEventListener('DOMContentLoaded', () => {
    new PitchAnalyzerApp();
});