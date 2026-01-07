import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TTS.api import TTS
from app.core.config import TTS_MODEL_PATH, TTS_CONFIG_PATH

print("MODEL:", TTS_MODEL_PATH, "exists:", os.path.exists(TTS_MODEL_PATH))
print("CONFIG:", TTS_CONFIG_PATH, "exists:", os.path.exists(TTS_CONFIG_PATH))

tts = TTS(
    model_path=TTS_MODEL_PATH,
    config_path=TTS_CONFIG_PATH,
    progress_bar=False,
    gpu=False,
)

tts.tts_to_file(text="你好，我是本地 TTS 测试。", file_path="test_local_tts.wav")
print("done -> test_local_tts.wav")
