# app/client/client.py
"""
语音助手客户端 v5 - USB 麦克风版本
适配 UGREEN CM379 USB Audio (双通道 16位)
"""
import asyncio
import json
import base64
import time
import sys
import threading
import subprocess
import os
import tempfile
from datetime import datetime

import websockets
try:
    from websockets.client import connect as ws_connect
except ImportError:
    from websockets import connect as ws_connect

from client_config import (
    SERVER_HOST, SERVER_PORT,
    AUTO_RECONNECT, RECONNECT_INTERVAL, MAX_RECONNECT_ATTEMPTS,
    SAMPLE_RATE, MIC_HW_ID, MIC_CHANNELS, MIC_FORMAT, MOCK_MODE
)

# 播放设备 (ReSpeaker 板载扬声器)
PLAYBACK_DEVICE = "plughw:3,0"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


class AudioManager:
    """音频管理器 - USB 麦克风版本"""
    
    def __init__(self):
        self.recording = False
        self.record_proc = None
        self.temp_file = None
        
        log(f"[音频] 录音设备: {MIC_HW_ID}")
        log(f"[音频] 录音格式: {MIC_CHANNELS}通道 {MIC_FORMAT}")
    
    def start_recording(self):
        """开始录音"""
        if self.recording:
            return
            
        log("[录音] 开始...")
        
        # 清理残留进程
        subprocess.run(["pkill", "-9", "arecord"], capture_output=True)
        time.sleep(0.1)
        
        # 临时文件
        self.temp_file = tempfile.mktemp(suffix=".wav")
        
        try:
            # USB 麦克风录制命令
            cmd = [
                "arecord",
                "-D", MIC_HW_ID,          # plughw:4,0
                "-f", MIC_FORMAT,          # S16_LE
                "-r", str(SAMPLE_RATE),    # 16000
                "-c", str(MIC_CHANNELS),   # 2
                "-t", "wav",               # WAV 格式
                "-q",                      # 安静模式
                self.temp_file
            ]
            
            log(f"[录音] 命令: {' '.join(cmd)}")
            self.record_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.recording = True
            log(f"[录音] PID={self.record_proc.pid}")
            
        except Exception as e:
            log(f"[录音] 启动失败: {e}")
            self.recording = False
    
    def stop_recording(self) -> bytes:
        """停止录音并返回 PCM 数据"""
        if not self.recording:
            return b''
        
        log("[录音] 停止...")
        self.recording = False
        audio_data = b''
        
        try:
            # 停止录音进程
            if self.record_proc:
                self.record_proc.terminate()
                try:
                    self.record_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.record_proc.kill()
                    self.record_proc.wait()
                self.record_proc = None
            
            time.sleep(0.2)
            
            # 读取录音文件并转换为单通道 PCM
            if self.temp_file and os.path.exists(self.temp_file):
                file_size = os.path.getsize(self.temp_file)
                log(f"[录音] WAV 文件: {file_size} bytes")
                
                # 使用 sox 转换: 双通道 -> 单通道 PCM
                pcm_file = tempfile.mktemp(suffix=".raw")
                
                result = subprocess.run([
                    "sox",
                    self.temp_file,                 # 输入 WAV
                    "-t", "raw",                    # 输出格式
                    "-r", str(SAMPLE_RATE),         # 采样率
                    "-b", "16",                     # 16 位
                    "-c", "1",                      # 单通道
                    "-e", "signed-integer",         # 有符号整数
                    pcm_file,                       # 输出文件
                    "remix", "1,2"                  # 混合两个通道
                ], capture_output=True, timeout=10)
                
                if result.returncode == 0 and os.path.exists(pcm_file):
                    with open(pcm_file, 'rb') as f:
                        audio_data = f.read()
                    os.remove(pcm_file)
                    
                    duration = len(audio_data) / SAMPLE_RATE / 2
                    log(f"[录音] PCM: {len(audio_data)} bytes ({duration:.1f}秒)")
                else:
                    log(f"[录音] sox 转换失败: {result.stderr.decode()}")
                
                # 清理临时文件
                os.remove(self.temp_file)
                self.temp_file = None
                
        except Exception as e:
            log(f"[录音] 错误: {e}")
            import traceback
            traceback.print_exc()
        
        return audio_data
    
    def play_audio(self, wav_data: bytes):
        """播放音频"""
        if not wav_data:
            return
            
        tmp_file = None
        try:
            tmp_file = tempfile.mktemp(suffix=".wav")
            with open(tmp_file, 'wb') as f:
                f.write(wav_data)
            
            log(f"[播放] 开始 ({len(wav_data)} bytes)")
            
            # 使用 ReSpeaker 播放设备
            proc = subprocess.Popen(
                ["aplay", "-D", PLAYBACK_DEVICE, "-q", tmp_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            proc.wait(timeout=60)
            log("[播放] 完成")
            
        except subprocess.TimeoutExpired:
            log("[播放] 超时")
        except Exception as e:
            log(f"[播放] 失败: {e}")
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass


class VoiceClient:
    """语音客户端"""
    
    def __init__(self):
        self.ws = None
        self.audio = AudioManager()
        self.is_recording = False
        self.running = True
        self.connected = False
        
    async def connect(self):
        """连接服务器"""
        uri = f"ws://{SERVER_HOST}:{SERVER_PORT}"
        log(f"连接 {uri}...")
        
        try:
            self.ws = await ws_connect(
                uri,
                max_size=20*1024*1024,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=30
            )
            self.connected = True
            log("✅ 已连接")
            return True
        except Exception as e:
            log(f"❌ 连接失败: {e}")
            return False
    
    async def send_audio(self, data: bytes):
        """发送音频数据"""
        if not self.ws or not data or not self.connected:
            return
            
        try:
            await self.ws.send(json.dumps({"type": "audio", "action": "start"}))
            
            # 分块发送
            chunk_size = 32 * 1024
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                await self.ws.send(json.dumps({
                    "type": "audio",
                    "action": "data",
                    "data": base64.b64encode(chunk).decode()
                }))
            
            await self.ws.send(json.dumps({"type": "audio", "action": "end"}))
            log("音频已发送")
            
        except Exception as e:
            log(f"发送失败: {e}")
            self.connected = False
    
    async def message_handler(self):
        """处理服务器消息"""
        try:
            async for msg in self.ws:
                if not self.connected:
                    break
                    
                try:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "asr_result":
                        text = data.get("text", "")
                        print(f"\n💬 识别: {text}")
                        
                    elif msg_type == "tts_audio":
                        text = data.get("text", "")
                        print(f"\n🔊 回复: {text}")
                        
                        if data.get("data"):
                            audio_bytes = base64.b64decode(data["data"])
                            threading.Thread(
                                target=self.audio.play_audio,
                                args=(audio_bytes,),
                                daemon=True
                            ).start()
                            
                    elif msg_type == "state":
                        state = data.get("state", "")
                        if state == "idle":
                            log("服务器处理完成")
                            
                except Exception as e:
                    log(f"处理消息错误: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            log(f"⚠️ 连接关闭: code={e.code}, reason='{e.reason}'")
        except Exception as e:
            log(f"❌ 消息处理错误: {e}")
            import traceback
            traceback.print_exc()
        
        self.connected = False
    
    def keyboard_thread(self, loop):
        """键盘输入监听"""
        while self.running:
            try:
                input()
                if self.connected:
                    asyncio.run_coroutine_threadsafe(
                        self.toggle_record(), loop
                    )
            except EOFError:
                break
            except:
                pass
    
    async def toggle_record(self):
        """切换录音状态"""
        if self.is_recording:
            print("\n⏹ 停止录音...")
            self.is_recording = False
            
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, self.audio.stop_recording)
            
            if len(audio_data) > 3200:
                await self.send_audio(audio_data)
            else:
                print("⚠️ 录音太短")
        else:
            print("\n🔴 开始录音... (按回车停止)")
            self.is_recording = True
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.audio.start_recording)
    
    async def run(self):
        """主循环"""
        loop = asyncio.get_event_loop()
        
        threading.Thread(
            target=self.keyboard_thread,
            args=(loop,),
            daemon=True
        ).start()
        
        retry_count = 0
        
        while self.running:
            if not await self.connect():
                retry_count += 1
                if retry_count >= MAX_RECONNECT_ATTEMPTS:
                    log("达到最大重试次数，退出")
                    break
                log(f"{RECONNECT_INTERVAL}秒后重试 ({retry_count}/{MAX_RECONNECT_ATTEMPTS})...")
                await asyncio.sleep(RECONNECT_INTERVAL)
                continue
            
            retry_count = 0
            
            print("\n" + "="*40)
            print("  🎤 按 [回车] 开始/停止录音")
            print("  🚪 按 [Ctrl+C] 退出")
            print("="*40 + "\n")
            
            await self.message_handler()
            
            log("连接断开")
            self.connected = False
            
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
            
            if self.running and AUTO_RECONNECT:
                log("准备重连...")
                await asyncio.sleep(2)
            else:
                break
        
        log("客户端退出")


def main():
    print("="*50)
    print("  语音助手客户端 v5")
    print("  (USB 麦克风版本)")
    print("="*50)
    
    client = VoiceClient()
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n退出")
    except Exception as e:
        log(f"❌ 程序崩溃: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.running = False
        log("程序结束")


if __name__ == "__main__":
    main()