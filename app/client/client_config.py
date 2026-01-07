# client_config.py
"""
客户端配置文件 (树莓派端)
"""

# ========== 服务器连接配置 ==========
SERVER_HOST = "127.0.0.1"  # 本地测试用 127.0.0.1，树莓派连接时改为 PC 的 IP (如 172.18.238.255)
SERVER_PORT = 8765             # 服务器端口

# 自动重连设置
AUTO_RECONNECT = True          # 断线自动重连
RECONNECT_INTERVAL = 5         # 重连间隔(秒)
MAX_RECONNECT_ATTEMPTS = 10    # 最大重连次数

# ========== 音频配置 ==========
SAMPLE_RATE = 16000            # 采样率
CHUNK_SIZE = 960               # 每帧样本数
MIC_DEVICE_INDEX = 0           # 麦克风设备索引

# ========== 硬件配置 ==========
ENABLE_LED = True              # 启用 LED 反馈
LED_PIN_BLUE = 17              # 蓝灯 GPIO (聆听状态)
LED_PIN_GREEN = 27             # 绿灯 GPIO (说话状态)

# ========== Mock 模式 ==========
MOCK_MODE = False              # 模拟模式 (用于调试)
