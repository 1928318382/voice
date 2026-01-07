import os
from enum import Enum, auto

# ========== 路径相关 ==========

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
ASR_FLAG_FILE = os.path.join(BASE_DIR, ".asr_download_ok")

# ========== ASR 配置 ==========

# 1. 远程仓库 ID (必须带 iic/ 前缀)
ASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 2. 本地文件夹名称 (去掉 / 避免路径错误)
_asr_local_name = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ASR_MODEL_PATH = os.path.join(MODELS_DIR, _asr_local_name)

# ========== TTS 配置 (保持不变) ==========

TTS_MODEL_DIR = os.path.join(MODELS_DIR, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
TTS_MODEL_PATH = os.path.join(TTS_MODEL_DIR, "model_file.pth")
TTS_CONFIG_PATH = os.path.join(TTS_MODEL_DIR, "config_local.json")

TTS_VOCODER_PATH = None
TTS_VOCODER_CONFIG_PATH = None
USE_PYTTXS3_FALLBACK = True

# ========== 语音增强配置 ==========

# 增强模式: 'none', 'basic', 'advanced', 'full'
ENHANCEMENT_MODE = "advanced"

# 降噪参数
NOISE_REDUCTION_STRENGTH = 0.8      # 降噪强度 (0-1)
MIN_VOICE_RATIO = 0.1               # 最小语音占比
AGC_TARGET_LEVEL = 0.8              # AGC目标电平
AGC_MAX_GAIN = 10.0                 # AGC最大增益

# VAD参数 (语音活动检测)
VAD_AGGRESSIVENESS = 2              # 0-3，越高越严格
VAD_FRAME_DURATION = 30             # 帧长度(ms)

# 增强模型路径
ENHANCEMENT_MODELS_DIR = os.path.join(MODELS_DIR, "enhancement")
RN_NOISE_MODEL_PATH = os.path.join(ENHANCEMENT_MODELS_DIR, "rnnoise")

# ========== 声纹识别配置 ==========

SPEAKER_RECOGNITION_ENABLED = True
SPEAKER_MODEL_TYPE = "ecapa_tdnn"         # ecapa_tdnn / cam_plus
SPEAKER_MODEL_SOURCE = "speechbrain"      # speechbrain / modelscope / local
SPEAKER_MODEL_PATH = os.path.join(MODELS_DIR, "speaker", "ecapa_tdnn")

# 模型参数
SPEAKER_EMBEDDING_DIM = 192               # ECAPA-TDNN输出维度
SPEAKER_SAMPLE_RATE = 16000               # 采样率

# 相似度参数
SPEAKER_SIMILARITY_THRESHOLD = 0.75       # 相似度阈值 (0.7-0.8推荐)
SPEAKER_SIMILARITY_METRIC = "cosine"      # cosine / euclidean / plda

# 用户管理
MAX_REGISTERED_SPEAKERS = 20              # 最大注册用户数
MIN_ENROLLMENT_SAMPLES = 3                # 最少注册样本数
# SPEAKER_DATABASE_PATH 已移至 DATA_DIR 部分 (第90行)

# 音频预处理
SPEAKER_CHUNK_DURATION = 3.0              # 音频分块长度(秒)
SPEAKER_OVERLAP_DURATION = 0.5            # 分块重叠长度(秒)
SPEAKER_MIN_AUDIO_LENGTH = 1.0            # 最短音频长度(秒)

MODEL_PATHS = {
    "asr_model": ASR_MODEL_PATH,
    "tts_model": TTS_MODEL_PATH,
    "enhancement_models": ENHANCEMENT_MODELS_DIR,
    "speaker_model": SPEAKER_MODEL_PATH,
}

# ========== 心知天气 API 配置 ==========

SENIVERSE_API_KEY = os.getenv("SENIVERSE_API_KEY", "SZLjOELaoNVV7tSr3")
SENIVERSE_BASE_URL = "https://api.seniverse.com/v3"

# ========== 其他配置 (保持不变) ==========

# ========== 数据文件 ==========
DATA_DIR = os.path.join(BASE_DIR, "data")
SPEAKER_DATABASE_PATH = os.path.join(DATA_DIR, "speaker_database.pkl")
# 原 speaker/database.pkl 改为 data/speaker_database.pkl 或者 models/speaker/database.pkl?
# 之前是 os.path.join(MODELS_DIR, "speaker", "database.pkl")
# 既然 speaker_root.txt 移到了 data/，database.pkl 也可以视为 data。
# 但为了保持原模型路径逻辑，先不动 path 逻辑，只确认 input files。
# 待会检查 usage。

LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-8RslfTjQR7sUxiwZDNQkOxLmdimQWhj0uaDjAEEAQdkhWWjW")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://metahk.zenymes.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen3-235B-A22B-FP8")

SAMPLE_RATE = 16000
CHUNK_SIZE = 960
MIC_DEVICE_INDEX = 0

class SystemState(Enum):
    INITIALIZING = auto()
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()