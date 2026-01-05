import emotional.extract_feats.librosa as lf
from emotional.utils import parse_opt
import os
import numpy as np
import emotional.models as models
import emotional.utils as utils
import tempfile # <-- 新增导入 tempfile 库
import soundfile as sf  # ← 就加这一行


class EmotionRecognizer:
    # ... (__init__ 方法保持不变) ..
    def __init__(self):
        self.config = utils.parse_opt()
        self.model  = models.load(self.config)
        print("[Emotion] 情感识别模块加载完毕 (真实模式)")

    def analyze(self, audio_data: bytes) -> str:
        if not audio_data:
            return "neutral"

        # 1. 内存转 wav 临时文件
        pcm = np.frombuffer(audio_data, dtype=np.int16)
        if pcm.size == 0:
            return "neutral"

        # 录到静音/极低能量时，特征可能出现 NaN，直接返回 neutral 更稳定
        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
        if not np.isfinite(rms) or rms < 1e-4:
            return "neutral"

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, pcm, self.config.sample_rate)
            tmp = f.name

        # 2. 特征 → 预测
        feat      = lf.get_data(self.config, tmp, train=False)
        feat      = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        pred_id   = int(self.model.predict(feat)[0])
        pred_lbl  = self.config.class_labels[pred_id]

        # 3. 清理
        os.remove(tmp)
        return pred_lbl
