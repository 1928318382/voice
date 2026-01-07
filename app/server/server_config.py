# server_config.py
"""
服务器端配置文件
"""
import os

# ========== 网络配置 ==========
SERVER_HOST = "0.0.0.0"  # 监听所有网卡，接受来自任何 IP 的连接
SERVER_PORT = 8765       # WebSocket 端口

# ========== 功能开关 ==========
ENABLE_ASR = True        # 语音识别
ENABLE_TTS = True        # 语音合成
ENABLE_LLM = True        # 大语言模型
ENABLE_EMOTION = True    # 情感识别
ENABLE_SPEAKER = True    # 声纹识别
ENABLE_ENHANCEMENT = True  # 语音增强

# ========== 功能模块开关 ==========
ENABLE_SCHEDULE = True   # 日程管理
ENABLE_WEATHER = True    # 天气查询
ENABLE_NEWS = True       # 新闻查询
ENABLE_FESTIVAL = True   # 节日提醒
ENABLE_MESSAGE_BOARD = True  # 留言板

# ========== 性能配置 ==========
MAX_CLIENTS = 5          # 最大同时连接数
AUDIO_BUFFER_SIZE = 2000 # 音频队列大小
CONNECTION_TIMEOUT = 30  # 连接超时(秒)

# ========== 日志配置 ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
