# mfa_analyzer.py
import os
import shutil
import subprocess
import tempfile
import textgrid

class MFA_Analyzer:
    def __init__(self, acoustic_model="english_us_arpa", dictionary="english_us_arpa"):
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        if shutil.which("mfa") is None:
            raise FileNotFoundError("MFA not found. `conda activate aligner` を確認してください。")

    def align_audio_file(self, audio_path: str):
        word = os.path.splitext(os.path.basename(audio_path))[0].lower()
        with tempfile.TemporaryDirectory() as temp_dir:
            # .labファイルと音声ファイルを配置
            lab_path = os.path.join(temp_dir, f"{word}.lab")
            shutil.copy(audio_path, temp_dir)
            with open(lab_path, "w") as f:
                f.write(word)

            # MFAアライメント実行
            command = [
                "mfa", "align", temp_dir, self.dictionary, self.acoustic_model,
                temp_dir, "--clean", "--overwrite"
            ]
            subprocess.run(command, check=True, capture_output=True, timeout=120)

            # TextGridファイルを読み込み
            tg_path = os.path.join(temp_dir, f"{word}.TextGrid")
            if not os.path.exists(tg_path):
                raise FileNotFoundError("TextGrid が生成されていません")

            tg = textgrid.TextGrid.fromFile(tg_path)
            phones = tg.getFirst("phones")
            return [
                {"phoneme": p.mark, "start": p.minTime, "end": p.maxTime}
                for p in phones if p.mark.lower() != "sil"
            ]