# client.py
"""
语音助手客户端 (树莓派端)
连接服务器，发送音频，接收并播放 TTS
"""
import asyncio
import json
import base64
import time
import sys
import threading
import argparse
import tempfile
import os
from enum import Enum, auto

import websockets
from websockets.client import connect

from client_config import (
    SERVER_HOST, SERVER_PORT,
    AUTO_RECONNECT, RECONNECT_INTERVAL, MAX_RECONNECT_ATTEMPTS,
    SAMPLE_RATE, CHUNK_SIZE, MIC_DEVICE_INDEX,
    ENABLE_LED, MOCK_MODE
)


class SystemState(Enum):
    """系统状态"""
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


class LEDController:
    """LED 控制器"""
    
    def __init__(self, mock=False):
        self.mock = mock or not ENABLE_LED
        if not self.mock:
            try:
                from gpiozero import LED
                from client_config import LED_PIN_BLUE, LED_PIN_GREEN
                self.led_blue = LED(LED_PIN_BLUE)
                self.led_green = LED(LED_PIN_GREEN)
            except Exception as e:
                print(f"[LED] GPIO 初始化失败: {e}，使用模拟模式")
                self.mock = True
    
    def set_state(self, state: SystemState):
        """根据状态设置 LED"""
        if self.mock:
            color_map = {
                SystemState.IDLE: "⚫ OFF",
                SystemState.LISTENING: "🔵 BLUE",
                SystemState.THINKING: "🟡 YELLOW",
                SystemState.SPEAKING: "🟢 GREEN",
                SystemState.ERROR: "🔴 RED"
            }
            print(f"  [LED] {color_map.get(state, 'UNKNOWN')}")
            return
        
        self.led_blue.off()
        self.led_green.off()
        
        if state == SystemState.LISTENING:
            self.led_blue.on()
        elif state == SystemState.SPEAKING:
            self.led_green.on()


class AudioDevice:
    """音频设备管理"""
    
    def __init__(self, mock=False):
        self.mock = mock
        self.pa = None
        self.stream = None
        
        if not mock:
            try:
                import pyaudio
                self.pa = pyaudio.PyAudio()
            except Exception as e:
                print(f"[Audio] PyAudio 初始化失败: {e}，使用模拟模式")
                self.mock = True
    
    def start_stream(self):
        """启动录音流"""
        if self.mock:
            print("[Audio] 模拟麦克风已启动")
            return
        
        try:
            import pyaudio
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=CHUNK_SIZE
            )
            print("[Audio] 麦克风已启动")
        except Exception as e:
            print(f"[Audio] 麦克风启动失败: {e}")
            self.mock = True
    
    def read_chunk(self) -> bytes:
        """读取一帧音频"""
        if self.mock:
            time.sleep(CHUNK_SIZE / SAMPLE_RATE)
            import numpy as np
            return np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        
        if self.stream:
            try:
                return self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except Exception as e:
                print(f"[Audio] 读取错误: {e}")
        return b''
    
    def stop_stream(self):
        """停止录音流"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
    
    def play_wav(self, wav_data: bytes):
        """播放 WAV 数据"""
        if self.mock:
            print("[Audio] 模拟播放音频...")
            time.sleep(1)
            return
        
        try:
            # 写入临时文件并播放
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                tmp_file = f.name
            
            try:
                if sys.platform.startswith("win"):
                    import winsound
                    winsound.PlaySound(tmp_file, winsound.SND_FILENAME)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["afplay", tmp_file], check=False)
                else:
                    import subprocess
                    subprocess.run(["aplay", tmp_file], check=False)
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    
        except Exception as e:
            print(f"[Audio] 播放失败: {e}")


class VoiceClient:
    """语音助手客户端"""
    
    def __init__(self, mock=False):
        self.mock = mock or MOCK_MODE
        self.running = True
        self.connected = False
        self.websocket = None
        self.state = SystemState.IDLE
        self.reconnect_count = 0
        
        # 硬件
        self.led = LEDController(mock=self.mock)
        self.audio = AudioDevice(mock=self.mock)
        
        # 录音控制
        self.is_recording = False
        self.audio_buffer = bytearray()
        
        # 命令队列
        self.cmd_queue = asyncio.Queue()
    
    def set_state(self, state: SystemState):
        """设置状态"""
        self.state = state
        self.led.set_state(state)
    
    async def connect_server(self):
        """连接服务器"""
        uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
        print(f"[Client] 正在连接服务器: {uri}")
        
        try:
            self.websocket = await connect(
                uri,
                max_size=10 * 1024 * 1024  # 10MB
            )
            self.connected = True
            self.reconnect_count = 0
            print("[Client] ✅ 已连接到服务器")
            return True
        except Exception as e:
            print(f"[Client] ❌ 连接失败: {e}")
            self.connected = False
            return False
    
    async def handle_messages(self):
        """处理服务器消息"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.process_server_message(data)
                except json.JSONDecodeError:
                    print("[Client] 收到无效消息")
        except websockets.exceptions.ConnectionClosed:
            print("[Client] 连接已断开")
            self.connected = False
    
    async def process_server_message(self, data: dict):
        """处理服务器消息"""
        msg_type = data.get("type", "")
        
        if msg_type == "connected":
            print(f"[Server] {data.get('message', '')}")
            
        elif msg_type == "state":
            state_str = data.get("state", "idle")
            state_map = {
                "idle": SystemState.IDLE,
                "listening": SystemState.LISTENING,
                "thinking": SystemState.THINKING,
                "speaking": SystemState.SPEAKING
            }
            self.set_state(state_map.get(state_str, SystemState.IDLE))
            
        elif msg_type == "asr_result":
            text = data.get("text", "")
            emotion = data.get("emotion", "neutral")
            speaker = data.get("speaker", "unknown")
            if text:
                print(f"\n💬 识别: {text}")
                if speaker != "unknown":
                    print(f"   👤 说话人: {speaker} | 😊 情绪: {emotion}")
            
        elif msg_type == "tts_audio":
            text = data.get("text", "")
            audio_b64 = data.get("data", "")
            
            print(f"\n🔊 回复: {text}")
            
            if audio_b64:
                wav_data = base64.b64decode(audio_b64)
                # 在新线程中播放，避免阻塞
                threading.Thread(
                    target=self.audio.play_wav, 
                    args=(wav_data,),
                    daemon=True
                ).start()
    
    async def send_audio_start(self):
        """发送录音开始信号"""
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "audio",
                "action": "start"
            }))
    
    async def send_audio_data(self, data: bytes):
        """发送音频数据"""
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "audio",
                "action": "data",
                "data": base64.b64encode(data).decode("utf-8")
            }))
    
    async def send_audio_end(self):
        """发送录音结束信号"""
        if self.websocket:
            await self.websocket.send(json.dumps({
                "type": "audio",
                "action": "end"
            }))
    
    def console_listener(self):
        """控制台监听线程"""
        while self.running:
            try:
                cmd = input()
                asyncio.run_coroutine_threadsafe(
                    self.cmd_queue.put(cmd.strip().lower()),
                    self.loop
                )
            except EOFError:
                break
    
    async def handle_commands(self):
        """处理键盘命令"""
        while self.running:
            try:
                cmd = await asyncio.wait_for(
                    self.cmd_queue.get(), 
                    timeout=0.05
                )
                
                if cmd == "q":
                    print("\n[Client] 正在退出...")
                    self.running = False
                    break
                else:
                    # 切换录音状态
                    if self.is_recording:
                        # 停止录音
                        print("\n✅ 录音结束，正在发送...")
                        self.is_recording = False
                        self.set_state(SystemState.THINKING)
                        await self.send_audio_end()
                    else:
                        # 开始录音
                        if not self.connected:
                            print("\n❌ 未连接到服务器")
                            continue
                        print("\n🔴 正在录音... (说完按回车)")
                        self.is_recording = True
                        self.set_state(SystemState.LISTENING)
                        self.audio_buffer.clear()
                        await self.send_audio_start()
                        
            except asyncio.TimeoutError:
                pass
    
    async def record_loop(self):
        """录音循环"""
        self.audio.start_stream()
        
        while self.running:
            if self.is_recording:
                chunk = self.audio.read_chunk()
                if chunk:
                    self.audio_buffer.extend(chunk)
                    # 每 10 帧发送一次 (约 0.6 秒)
                    if len(self.audio_buffer) >= CHUNK_SIZE * 2 * 10:
                        await self.send_audio_data(bytes(self.audio_buffer))
                        self.audio_buffer.clear()
            else:
                await asyncio.sleep(0.01)
        
        self.audio.stop_stream()
    
    async def run(self):
        """主运行循环"""
        self.loop = asyncio.get_event_loop()
        
        print("=" * 50)
        print("  语音助手客户端 (树莓派)")
        print("  [回车键] 切换录音/停止")
        print("  [q] + 回车 退出")
        print("=" * 50)
        
        # 启动控制台监听线程
        console_thread = threading.Thread(target=self.console_listener, daemon=True)
        console_thread.start()
        
        while self.running:
            # 连接服务器
            if not await self.connect_server():
                if AUTO_RECONNECT and self.reconnect_count < MAX_RECONNECT_ATTEMPTS:
                    self.reconnect_count += 1
                    print(f"[Client] {RECONNECT_INTERVAL} 秒后重试 ({self.reconnect_count}/{MAX_RECONNECT_ATTEMPTS})...")
                    await asyncio.sleep(RECONNECT_INTERVAL)
                    continue
                else:
                    print("[Client] 无法连接服务器，退出")
                    break
            
            self.set_state(SystemState.IDLE)
            print("\n[Client] 就绪，按回车开始对话...")
            
            # 启动任务
            try:
                await asyncio.gather(
                    self.handle_messages(),
                    self.handle_commands(),
                    self.record_loop()
                )
            except Exception as e:
                print(f"[Client] 错误: {e}")
            
            if not self.running:
                break
            
            # 断线重连
            if AUTO_RECONNECT:
                print("[Client] 尝试重新连接...")
                await asyncio.sleep(RECONNECT_INTERVAL)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音助手客户端")
    parser.add_argument("--mock", action="store_true", help="模拟模式")
    parser.add_argument("--host", type=str, help="服务器地址")
    parser.add_argument("--port", type=int, help="服务器端口")
    args = parser.parse_args()
    
    # 覆盖配置
    if args.host:
        import client_config
        client_config.SERVER_HOST = args.host
    if args.port:
        import client_config
        client_config.SERVER_PORT = args.port
    
    client = VoiceClient(mock=args.mock)
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n[Client] 已退出")


if __name__ == "__main__":
    main()
