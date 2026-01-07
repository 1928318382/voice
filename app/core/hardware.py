import time
import numpy as np
from app.core.config import SystemState, SAMPLE_RATE, CHUNK_SIZE, MIC_DEVICE_INDEX


class LEDController:
    """控制 ReSpeaker 像素环或 GPIO LED 灯光反馈"""

    def __init__(self, mock=False):
        self.mock = mock
        if not mock:
            try:
                from gpiozero import LED
                # 示例: GPIO 17 控制蓝色(听), 27 控制绿色(说)
                self.led_blue = LED(17)
                self.led_green = LED(27)
            except ImportError:
                print("[HW] GPIO库未安装，自动切换至模拟LED模式")
                self.mock = True
            except Exception as e:
                print(f"[HW] GPIO 初始化失败: {e}，切换至模拟模式")
                self.mock = True

    def set_state(self, state):
        """根据系统状态改变灯光颜色"""
        if self.mock:
            color_map = {
                SystemState.IDLE: "OFF (熄灭)",
                SystemState.LISTENING: "BLUE (蓝灯-聆听)",
                SystemState.THINKING: "YELLOW (黄灯-思考)",
                SystemState.SPEAKING: "GREEN (绿灯-说话)",
                SystemState.ERROR: "RED (红灯-错误)"
            }
            # 仅在状态变化时打印，避免刷屏 (逻辑在主程序控制)
            print(f"  [LED] 状态灯切换: {color_map.get(state, 'UNKNOWN')}")
            return

        # 真实硬件控制逻辑
        # 先关闭所有
        self.led_blue.off()
        self.led_green.off()

        if state == SystemState.LISTENING:
            self.led_blue.on()
        elif state == SystemState.SPEAKING:
            self.led_green.on()
        # THINKING 状态可以做呼吸灯效果，此处省略


class AudioDevice:
    """音频输入输出管理"""

    def __init__(self, mock=False):
        self.mock = mock
        self.pa = None
        self.sd = None
        self.stream = None
        self.backend = None
        if not mock:
            try:
                import pyaudio
                self.pa = pyaudio.PyAudio()
                self.backend = "pyaudio"
            except Exception:
                try:
                    import sounddevice as sd
                    self.sd = sd
                    self.backend = "sounddevice"
                except Exception:
                    print("[Audio] 未检测到 pyaudio/sounddevice，切换到 mock 模式")
                    self.mock = True

    def start_stream(self):
        """开启麦克风录音流"""
        if self.mock:
            print("[Audio] 虚拟麦克风已启动")
            return

        try:
            if self.backend == "pyaudio":
                import pyaudio
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=MIC_DEVICE_INDEX,
                    frames_per_buffer=CHUNK_SIZE
                )
            elif self.backend == "sounddevice":
                self.stream = self.sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    device=MIC_DEVICE_INDEX,
                    blocksize=CHUNK_SIZE,
                )
                self.stream.start()
            else:
                self.mock = True
                print("[Audio] 未选择可用后端，切换到 mock 模式")
        except Exception as e:
            print(f"[Audio] 麦克风打开失败: {e}，请检查 MIC_DEVICE_INDEX")
            self.mock = True

    def read_chunk(self):
        """读取一帧音频数据"""
        if self.mock:
            time.sleep(CHUNK_SIZE / SAMPLE_RATE)  # 模拟真实采样率延迟
            # 返回静音数据 (全0)，实际开发中可在这里注入测试音频文件
            return np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        if self.stream:
            try:
                if self.backend == "pyaudio":
                    return self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if self.backend == "sounddevice":
                    data, _ = self.stream.read(CHUNK_SIZE)
                    return data.tobytes()
            except Exception as e:
                print(f"[Audio] 读取错误: {e}")
                return b''
        return b''

    def play_audio(self, audio_data):
        """播放音频数据"""
        if self.mock:
            # 模拟播放耗时: 假设 audio_data 长度/采样率/2字节 = 秒数
            duration = len(audio_data) / SAMPLE_RATE / 2
            print(f"  [Audio] 播放音频中... ({duration:.2f}s)")
            time.sleep(duration)
            return

        # 使用 sounddevice 或 pyaudio 播放
        try:
            import sounddevice as sd
            # 假设音频是 int16 格式
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_np, SAMPLE_RATE)
            sd.wait()
        except ImportError:
            # 降级方案
            pass
