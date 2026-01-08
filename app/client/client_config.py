# app/client/client_config.py
"""
客户端配置文件 (树莓派 8-Mic 适配版)
"""

# ========== 服务器连接配置 ==========
# ⚠️⚠️⚠️ 请务必修改这里为你的电脑 IP (例如 192.168.31.206)
# 如果填 127.0.0.1 树莓派是连不上电脑的
SERVER_HOST = "172.18.238.255"  
SERVER_PORT = 8765

# ========== 自动重连配置 ==========
AUTO_RECONNECT = True          # 开启自动重连
RECONNECT_INTERVAL = 5         # 重连间隔(秒)
MAX_RECONNECT_ATTEMPTS = 100   # 最大尝试次数

# ========== 音频配置 ==========
SAMPLE_RATE = 16000
CHUNK_SIZE = 960               # 每帧大小 (约60ms)

# USB 麦克风配置 (UGREEN CM379)
MIC_HW_ID = "plughw:4,0"       # USB 麦克风
MIC_CHANNELS = 2               # USB 麦克风是双通道
MIC_FORMAT = "S16_LE"          # 16位格式

# ========== 硬件配置 ==========
# 8-Mic 灯环需要特殊驱动，暂时关闭以防报错
ENABLE_LED = False             
LED_PIN_BLUE = 17 
LED_PIN_GREEN = 27

# ========== Mock 模式 ==========
MOCK_MODE = False              # 关闭模拟模式