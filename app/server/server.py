# server.py
"""
语音助手服务器端 (PC)
使用 WebSocket 接收客户端音频，处理后返回 TTS 音频
"""
import asyncio
import json
import base64
import time
import warnings
import threading
import queue
from typing import Optional, Dict, Any

# 忽略 jieba 的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import websockets
# 尝试使用新版 API (asyncio) 避免弃用警告，如果失败则回退
try:
    from websockets.asyncio.server import serve
except ImportError:
    from websockets.server import serve

# 添加项目根目录到 sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server_config import (
    SERVER_HOST, SERVER_PORT, MAX_CLIENTS,
    ENABLE_ASR, ENABLE_TTS, ENABLE_LLM,
    ENABLE_EMOTION, ENABLE_SPEAKER, ENABLE_ENHANCEMENT,
    ENABLE_SCHEDULE, ENABLE_WEATHER, ENABLE_NEWS, 
    ENABLE_FESTIVAL, ENABLE_MESSAGE_BOARD
)
from app.core.config import SystemState, SAMPLE_RATE

# 导入处理引擎
import numpy as np
import tempfile
import soundfile as sf
import os


class VoiceServer:
    """语音助手服务器"""
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}  # 连接的客户端
        self.running = True
        
        # ===== 初始化各处理模块 =====
        print("[Server] 正在初始化服务器模块...")
        
        # 情感识别
        self.emotion_engine = None
        if ENABLE_EMOTION:
            try:
                from app.core.emotion import EmotionRecognizer
                self.emotion_engine = EmotionRecognizer()
                print("[Server] ✅ 情感识别模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 情感识别模块加载失败: {e}")
        
        # 语音增强
        self.audio_enhancer = None
        if ENABLE_ENHANCEMENT:
            try:
                from app.core.enhancement import AudioEnhancer
                self.audio_enhancer = AudioEnhancer()
                print("[Server] ✅ 语音增强模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 语音增强模块加载失败: {e}")
        
        # 声纹识别
        # 声纹识别
        self.speaker_recognizer = None
        if ENABLE_SPEAKER:
            try:
                from app.core.speaker import ECAPATDNNRecognizer
                self.speaker_recognizer = ECAPATDNNRecognizer()
                print("[Server] ✅ 声纹识别模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 声纹识别模块加载失败: {e}")
        
        # ASR 模型
        self.asr_model = None
        if ENABLE_ASR:
            try:
                from funasr import AutoModel
                from app.core.config import ASR_MODEL_PATH
                self.asr_model = AutoModel(
                    model=ASR_MODEL_PATH,
                    device="cpu",
                    disable_update=True,
                    disable_log=True,
                    trust_remote_code=True
                )
                print("[Server] ✅ ASR 模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ ASR 模块加载失败: {e}")
        
        # TTS 引擎
        self.tts_engine = None
        if ENABLE_TTS:
            try:
                from TTS.api import TTS as CoquiTTS
                from app.core.config import TTS_MODEL_PATH, TTS_CONFIG_PATH
                self.tts_engine = CoquiTTS(
                    model_path=TTS_MODEL_PATH,
                    config_path=TTS_CONFIG_PATH,
                    progress_bar=False,
                    gpu=False
                )
                print("[Server] ✅ TTS 模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ TTS 模块加载失败: {e}")
        
        # LLM 客户端
        self.llm_client = None
        if ENABLE_LLM:
            try:
                from openai import OpenAI
                from app.core.config import LLM_API_KEY, LLM_BASE_URL
                self.llm_client = OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL
                )
                print("[Server] ✅ LLM 模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ LLM 模块加载失败: {e}")
        
        # 功能处理器
        self.schedule_handler = None
        self.news_handler = None
        self.weather_handler = None
        self.festival_handler = None
        self.message_board_handler = None
        
        if ENABLE_SCHEDULE:
            try:
                from app.features import ScheduleCommandHandler
                self.schedule_handler = ScheduleCommandHandler()
                print("[Server] ✅ 日程管理模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 日程管理模块加载失败: {e}")
        
        if ENABLE_NEWS:
            try:
                from app.features import NewsCommandHandler
                self.news_handler = NewsCommandHandler()
                print("[Server] ✅ 新闻模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 新闻模块加载失败: {e}")
        
        if ENABLE_WEATHER:
            try:
                from app.features import WeatherCommandHandler
                self.weather_handler = WeatherCommandHandler()
                print("[Server] ✅ 天气模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 天气模块加载失败: {e}")
        
        if ENABLE_FESTIVAL:
            try:
                from app.features import FestivalCommandHandler
                self.festival_handler = FestivalCommandHandler()
                print("[Server] ✅ 节日提醒模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 节日提醒模块加载失败: {e}")
        
        if ENABLE_MESSAGE_BOARD and self.speaker_recognizer:
            try:
                from app.features import MessageBoardCommandHandler
                self.message_board_handler = MessageBoardCommandHandler(self.speaker_recognizer)
                print("[Server] ✅ 留言板模块加载完成")
            except Exception as e:
                print(f"[Server] ⚠️ 留言板模块加载失败: {e}")
        
        print("[Server] 服务器初始化完成!")

    # ==================== ASR 处理 ====================
    
    def process_asr(self, audio_data: bytes) -> Dict[str, Any]:
        """处理音频，返回识别结果"""
        result = {
            "text": "",
            "emotion": "neutral",
            "speaker": "unknown"
        }
        
        if not audio_data or len(audio_data) < 1920:  # 至少 0.06 秒
            return result
        
        # 语音增强
        if self.audio_enhancer:
            try:
                audio_data = self.audio_enhancer.process(audio_data)
            except Exception as e:
                print(f"[Server] 语音增强失败: {e}")
        
        # 情感识别
        if self.emotion_engine:
            try:
                result["emotion"] = self.emotion_engine.analyze(audio_data)
            except Exception as e:
                print(f"[Server] 情感识别失败: {e}")
        
        # 声纹识别
        if self.speaker_recognizer:
            try:
                result["speaker"] = self.speaker_recognizer.identify(audio_data)
            except Exception as e:
                print(f"[Server] 声纹识别失败: {e}")
        
        # ASR 识别
        if self.asr_model:
            try:
                # 转换为 numpy 数组
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # 写入临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio_np, SAMPLE_RATE)
                    tmp_file = f.name
                
                try:
                    res = self.asr_model.generate(
                        input=tmp_file, 
                        batch_size_s=300, 
                        disable_pbar=True
                    )
                    
                    if isinstance(res, list) and len(res) > 0:
                        result["text"] = res[0].get("text", "")
                    elif isinstance(res, dict):
                        result["text"] = res.get("text", "")
                    
                    # 清理空格
                    result["text"] = result["text"].replace(" ", "").strip()
                    
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                        
            except Exception as e:
                print(f"[Server] ASR 识别失败: {e}")
        
        return result

    # ==================== LLM 处理 ====================
    
    def process_llm(self, text: str, emotion: str = "neutral", speaker: str = "unknown") -> str:
        """调用 LLM 生成回复"""
        if not self.llm_client:
            return "抱歉，语言模型未就绪。"
        
        try:
            from app.core.config import LLM_MODEL_NAME
            
            speaker_info = f"说话人：{speaker}。" if speaker != "unknown" else ""
            system_prompt = (
                "你是一个基于树莓派的智能助手'小语'。"
                f"用户当前情绪：{emotion}。"
                f"{speaker_info}"
                "请用简短、亲切的中文回复（50字以内）。"
                "不要使用Markdown格式，直接输出纯文本。"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[Server] LLM 调用失败: {e}")
            return "抱歉，我的大脑连接有点问题。"

    # ==================== TTS 处理 ====================
    
    def process_tts(self, text: str) -> Optional[bytes]:
        """将文本转换为语音，返回 WAV 数据"""
        if not self.tts_engine or not text:
            return None
        
        try:
            import re
            
            # 文本清洗
            text = re.sub(r"[A-Za-z0-9]+", " ", text)
            text = re.sub(r"[^\u4e00-\u9fff，。！？、；：,.!?…~\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            
            if not text:
                return None
            
            # 添加句号
            if not text.endswith(("。", "！", "？", ".", "!", "?", "…", "~")):
                text += "。"
            
            # 合成到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_file = f.name
            
            try:
                self.tts_engine.tts_to_file(text=text, file_path=tmp_file)
                
                # 读取 WAV 数据
                with open(tmp_file, "rb") as f:
                    wav_data = f.read()
                
                return wav_data
                
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    
        except Exception as e:
            print(f"[Server] TTS 合成失败: {e}")
            return None

    # ==================== 功能处理 ====================
    
    def handle_features(self, text: str, speaker: str = "unknown") -> Optional[str]:
        """处理功能命令，返回回复文本或 None"""
        
        # 留言板
        if self.message_board_handler:
            reply = self.message_board_handler.handle(text, speaker)
            if reply:
                return reply
        
        # 日程管理
        if self.schedule_handler:
            reply = self.schedule_handler.handle(text)
            if reply:
                # 处理 PARTIAL_QUERY 前缀
                if reply.startswith("PARTIAL_QUERY:"):
                    parts = reply.split(":", 3)
                    if len(parts) >= 4:
                        return parts[1]  # 返回语音文本部分
                return reply
        
        # 天气查询
        if self.weather_handler:
            reply = self.weather_handler.handle(text)
            if reply:
                return reply
        
        # 新闻查询
        if self.news_handler:
            reply = self.news_handler.handle(text)
            if reply:
                return reply
        
        # 节日提醒
        if self.festival_handler:
            reply = self.festival_handler.handle(text)
            if reply:
                return reply
        
        return None

    # ==================== WebSocket 处理 ====================
    
    async def handle_client(self, websocket):
        """处理单个客户端连接"""
        client_id = id(websocket)
        client_addr = websocket.remote_address
        print(f"[Server] 客户端连接: {client_addr} (ID: {client_id})")
        
        self.clients[client_id] = {
            "websocket": websocket,
            "audio_buffer": bytearray(),
            "state": SystemState.IDLE
        }
        
        try:
            # 发送连接确认
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "欢迎连接语音助手服务器"
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(client_id, data)
                except json.JSONDecodeError:
                    print(f"[Server] 无效的 JSON 消息")
                except Exception as e:
                    print(f"[Server] 处理消息错误: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"[Server] 客户端断开: {client_addr}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def process_message(self, client_id: str, data: Dict[str, Any]):
        """处理客户端消息"""
        msg_type = data.get("type", "")
        client = self.clients.get(client_id)
        
        if not client:
            return
        
        websocket = client["websocket"]
        
        if msg_type == "audio":
            action = data.get("action", "")
            
            if action == "start":
                # 开始录音
                client["audio_buffer"] = bytearray()
                client["state"] = SystemState.LISTENING
                await self.send_state(websocket, "listening")
                print(f"[Server] 客户端 {client_id} 开始录音")
                
            elif action == "data":
                # 接收音频数据
                audio_b64 = data.get("data", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    client["audio_buffer"].extend(audio_bytes)
                    
            elif action == "end":
                # 录音结束，开始处理
                print(f"[Server] 客户端 {client_id} 录音结束，开始处理...")
                client["state"] = SystemState.THINKING
                await self.send_state(websocket, "thinking")
                
                audio_data = bytes(client["audio_buffer"])
                client["audio_buffer"] = bytearray()
                
                # 异步处理 (避免阻塞)
                asyncio.create_task(
                    self.process_and_respond(websocket, audio_data)
                )
    
    async def process_and_respond(self, websocket, audio_data: bytes):
        """处理音频并返回响应"""
        try:
            # 1. ASR 识别
            asr_result = await asyncio.get_event_loop().run_in_executor(
                None, self.process_asr, audio_data
            )
            
            text = asr_result.get("text", "")
            emotion = asr_result.get("emotion", "neutral")
            speaker = asr_result.get("speaker", "unknown")
            
            print(f"[Server] ASR 结果: {text} (情感: {emotion}, 说话人: {speaker})")
            
            # 发送 ASR 结果
            await websocket.send(json.dumps({
                "type": "asr_result",
                "text": text,
                "emotion": emotion,
                "speaker": speaker
            }))
            
            if not text:
                await self.send_state(websocket, "idle")
                return
            
            # 2. 检查功能命令
            feature_reply = await asyncio.get_event_loop().run_in_executor(
                None, self.handle_features, text, speaker
            )
            
            if feature_reply:
                reply_text = feature_reply
            else:
                # 3. 调用 LLM
                reply_text = await asyncio.get_event_loop().run_in_executor(
                    None, self.process_llm, text, emotion, speaker
                )
            
            print(f"[Server] 回复: {reply_text}")
            
            # 4. TTS 合成
            await self.send_state(websocket, "speaking")
            
            wav_data = await asyncio.get_event_loop().run_in_executor(
                None, self.process_tts, reply_text
            )
            
            if wav_data:
                # 发送 TTS 音频
                await websocket.send(json.dumps({
                    "type": "tts_audio",
                    "text": reply_text,
                    "data": base64.b64encode(wav_data).decode("utf-8"),
                    "is_final": True
                }))
            
            await self.send_state(websocket, "idle")
            
        except Exception as e:
            print(f"[Server] 处理错误: {e}")
            await self.send_state(websocket, "idle")
    
    async def send_state(self, websocket, state: str):
        """发送状态更新"""
        try:
            await websocket.send(json.dumps({
                "type": "state",
                "state": state
            }))
        except Exception:
            pass
    
    async def start(self):
        """启动服务器"""
        print("=" * 50)
        print(f"  语音助手服务器启动")
        print(f"  地址: ws://{SERVER_HOST}:{SERVER_PORT}")
        print(f"  按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        async with serve(
            self.handle_client, 
            SERVER_HOST, 
            SERVER_PORT,
            max_size=10 * 1024 * 1024  # 10MB 最大消息
        ):
            await asyncio.Future()  # 永久运行


def main():
    """主函数"""
    server = VoiceServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[Server] 服务器已停止")


if __name__ == "__main__":
    main()
