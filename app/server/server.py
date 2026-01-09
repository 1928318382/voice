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
            print(f"[Server] ⚠️ 接收到的音频数据太短: {len(audio_data)} bytes")
            return result
        
        print(f"[Server] 接收到音频数据: {len(audio_data)} bytes")
        
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
                
                print(f"[Server]DEBUG ASR临时文件已保存: {tmp_file}, 数据形状: {audio_np.shape}, 采样率: {SAMPLE_RATE}")

                try:
                    res = self.asr_model.generate(
                        input=tmp_file, 
                        batch_size_s=300, 
                        disable_pbar=True
                    )
                    print(f"[Server]DEBUG ASR原始结果: {res}")
                    
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
                import traceback
                traceback.print_exc()
        
        return result

    # ==================== LLM 处理 ====================
    
    def _generate_chat_response(self, text: str, emotion: str = "neutral", speaker: str = "unknown") -> str:
        """调用 LLM 生成 闲聊 回复 (原 process_llm)"""
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

    def classify_intent(self, user_text: str) -> str:
        """使用LLM对用户输入进行意图分类"""
        if not self.llm_client:
            print("[Server] LLM未初始化，无法进行意图分类，默认chat")
            return "chat"

        from app.core.config import LLM_MODEL_NAME
        
        classification_prompt = (
            "你是一个意图分类助手。请仔细分析用户的输入，准确判断用户想要执行什么操作。\n\n"
            "可选的功能类型及判断标准：\n\n"
            "1. schedule（日程管理）：\n"
            "   - 明确包含：添加、记录、提醒、安排、查看、查询、删除、修改日程等动作\n"
            "   - 包含时间信息：明天、后天、早上、晚上、几点等\n"
            "   - 包含日程相关词：日程、提醒、安排、吃药、睡觉、起床、待办、任务等\n"
            "   - 示例：\"记一下明天早上8点开会\"、\"帮我提醒晚上10点睡觉\"、\"查看我的日程\"、\"删除编号3的日程\"、\"明天下午3点写报告\"\n"
            "   - 注意：如果包含\"纪念日\"、\"生日\"、\"节日\"等词，应该归类为festival，而不是schedule\n\n"
            "2. weather（天气查询）：\n"
            "   - 明确包含\"天气\"关键词\n"
            "   - 包含城市名称和天气相关词\n"
            "   - 示例：\"今天天气怎么样\"、\"北京未来三天天气\"、\"上海天气\"、\"明天会下雨吗\"\n\n"
            "3. news（新闻查询）：\n"
            "   - 明确包含：新闻、小贴士、建议、tip等关键词\n"
            "   - 示例：\"有什么新闻\"、\"看新闻\"、\"生活小贴士\"、\"职场建议\"、\"给我一些生活建议\"\n\n"
            "4. festival（节日提醒）：\n"
            "   - 明确包含：节日、节日提醒、添加节日、纪念日、生日、周年纪念等\n"
            "   - 包含\"纪念日\"、\"生日\"、\"节日\"等关键词，且用户想要设定或添加\n"
            "   - 示例：\"有哪些节日\"、\"添加节日\"、\"节日提醒\"、\"什么时候是春节\"、\"把一月八号设定为我的入团纪念日\"、\"添加我的生日\"、\"设定纪念日\"\n"
            "   - 注意：如果用户说\"设定XX纪念日\"、\"添加XX节日\"，应该归类为festival，而不是schedule\n\n"
            "5. message_board（留言板）：\n"
            "   - 明确包含：留言、查看留言、给XX留言等\n"
            "   - 示例：\"查看留言\"、\"给张三留言你好\"、\"我的留言\"、\"有留言吗\"\n\n"
            "6. chat（正常聊天）：\n"
            "   - 普通对话、问候、提问、闲聊、知识问答等\n"
            "   - 不涉及上述任何功能操作\n"
            "   - 示例：\"你好\"、\"今天心情不错\"、\"给我讲个笑话\"、\"什么是人工智能\"、\"谢谢\"、\"再见\"\n\n"
            "重要判断规则：\n"
            "- 必须明确包含功能相关的关键词或动作，才返回功能类型\n"
            "- 如果只是提到相关词但没有明确的操作意图，返回chat（例如：\"今天天气真好\"是聊天，不是查询天气）\n"
            "- 如果同时包含功能意图和聊天内容，优先返回功能类型\n"
            "- 如果无法确定或模糊不清，返回chat\n"
            "- 问候语、感谢、告别等社交用语，返回chat\n\n"
            "请只返回一个单词：schedule、weather、news、festival、message_board 或 chat，不要返回其他内容，不要解释。"
        )

        messages = [
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": user_text}
        ]

        try:
            # 简化版超时处理，server端通常网络较好，或者直接依赖 client timeout
            # 这里简单设置 API timeout
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=10,
                timeout=5.0
            )
            
            intent = response.choices[0].message.content.strip().lower()
            intent = intent.replace("。", "").replace(".", "").replace("\n", "").strip()
            
            valid_intents = ["schedule", "weather", "news", "festival", "message_board", "chat"]
            if intent not in valid_intents:
                print(f"[Server] 分类结果无效: {intent}，默认返回chat")
                return "chat"
            
            return intent
            
        except Exception as e:
            print(f"[Server] 意图分类失败: {e}，默认返回chat")
            return "chat"


    # ==================== TTS 处理 (Coqui TTS) ====================
    
    # 句子最大长度（字符数），超过则分段
    MAX_SENTENCE_LENGTH = 80
    
    def _normalize_text(self, text: str) -> str:
        """清洗文本以减少 TTS 词表缺失导致的异常发音。"""
        import re
        if not text:
            return ""
        
        # Convert newlines to sentence delimiters to preserve list structure
        text = re.sub(r'\n+', '。', text)
        
        # 先处理日期
        text = self._convert_dates_in_text(text)

        # 将数字转换为中文读法
        text = self._convert_numbers_in_text(text)
        
        # 清理不常见符号 (保留字母)
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z，。！？、；：,.!?…~\s]", " ", text)
        # 统一空白
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _convert_dates_in_text(self, text: str) -> str:
        """将日期格式 (如 2026-1-7, 2026/01/07) 转换为中文读法"""
        import re
        CHINESE_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        
        def replace_date(match):
            year = match.group(1)
            month = match.group(2)
            day = match.group(3)
            
            # 年份逐位读
            year_chinese = "".join(CHINESE_DIGITS[int(d)] for d in year)
            
            # 月份和日期按整数读
            month_chinese = self._integer_to_chinese(month)
            day_chinese = self._integer_to_chinese(day)
            
            return f"{year_chinese}年{month_chinese}月{day_chinese}日"

        pattern = r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:日)?'
        return re.sub(pattern, replace_date, text)
    
    def _convert_numbers_in_text(self, text: str) -> str:
        """将文本中的数字转换为中文读法"""
        import re
        def replace_number(match):
            num = match.group(0)
            return self._number_to_chinese(num)
        
        pattern = r'(?<!\d)-?\d+\.?\d*'
        return re.sub(pattern, replace_number, text)
    
    def _number_to_chinese(self, num_str: str) -> str:
        """将数字字符串转换为中文读法"""
        CHINESE_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        
        if not num_str:
            return ""
        
        # 处理小数
        if '.' in num_str:
            parts = num_str.split('.', 1)
            integer_part = self._integer_to_chinese(parts[0])
            decimal_part = ''.join(CHINESE_DIGITS[int(d)] for d in parts[1] if d.isdigit())
            return f"{integer_part}点{decimal_part}" if decimal_part else integer_part
        
        return self._integer_to_chinese(num_str)
    
    def _integer_to_chinese(self, num_str: str) -> str:
        """将整数字符串转换为中文"""
        CHINESE_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        CHINESE_UNITS = ['', '十', '百', '千']
        CHINESE_BIG_UNITS = ['', '万', '亿', '兆']
        
        if not num_str:
            return ""
        
        # 去除前导零
        num_str = num_str.lstrip('0') or '0'
        
        if num_str == '0':
            return '零'
        
        # 处理负数
        if num_str.startswith('-'):
            return '负' + self._integer_to_chinese(num_str[1:])
        
        # 对于特别长的数字（如电话号码），逐位读出
        if len(num_str) > 8:
            return ''.join(CHINESE_DIGITS[int(d)] for d in num_str if d.isdigit())
        
        result = []
        
        # 按4位一组处理
        groups = []
        while num_str:
            groups.append(num_str[-4:])
            num_str = num_str[:-4]
        groups.reverse()
        
        for group_idx, group in enumerate(groups):
            group_result = self._four_digits_to_chinese(group)
            if group_result:
                big_unit_idx = len(groups) - group_idx - 1
                if big_unit_idx < len(CHINESE_BIG_UNITS):
                    result.append(group_result + CHINESE_BIG_UNITS[big_unit_idx])
                else:
                    result.append(group_result)
        
        return ''.join(result)
    
    def _four_digits_to_chinese(self, num_str: str) -> str:
        """将4位以内的数字转换为中文"""
        CHINESE_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        CHINESE_UNITS = ['', '十', '百', '千']
        
        if not num_str:
            return ""
        
        num_str = num_str.lstrip('0') or '0'
        if num_str == '0':
            return ''
        
        result = []
        length = len(num_str)
        has_zero = False
        
        for i, digit in enumerate(num_str):
            d = int(digit)
            unit_idx = length - i - 1
            
            if d == 0:
                has_zero = True
            else:
                if has_zero:
                    result.append('零')
                    has_zero = False
                # 特殊处理"一十"读作"十"
                if not (d == 1 and unit_idx == 1 and i == 0):
                    result.append(CHINESE_DIGITS[d])
                if unit_idx < len(CHINESE_UNITS):
                    result.append(CHINESE_UNITS[unit_idx])
        
        return ''.join(result)
    
    def _split_long_text(self, text: str) -> list:
        """将长文本按句子边界分割，确保每段不超过 MAX_SENTENCE_LENGTH"""
        import re
        
        # 先按标点分句
        sentence_delimiters = r'([。！？!?…~；;]|\n)'
        parts = re.split(sentence_delimiters, text)
        
        # 重新组合：[句子, 标点, 句子, 标点, ...]
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if not part:
                continue
            if re.match(sentence_delimiters, part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current = part
        if current.strip():
            sentences.append(current.strip())
        
        # 合并短句，拆分超长句
        result = []
        buffer = ""
        
        for sentence in sentences:
            # 如果单个句子就超长，按逗号进一步拆分
            if len(sentence) > self.MAX_SENTENCE_LENGTH:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                # 按逗号拆分
                sub_sentences = re.split(r'([，,、])', sentence)
                sub_buffer = ""
                for sub in sub_sentences:
                    if len(sub_buffer) + len(sub) <= self.MAX_SENTENCE_LENGTH:
                        sub_buffer += sub
                    else:
                        if sub_buffer:
                            result.append(sub_buffer)
                        sub_buffer = sub
                if sub_buffer:
                    result.append(sub_buffer)
            elif len(buffer) + len(sentence) <= self.MAX_SENTENCE_LENGTH:
                buffer += sentence
            else:
                if buffer:
                    result.append(buffer)
                buffer = sentence
        
        if buffer:
            result.append(buffer)
        
        return result if result else [text]
    
    def _postprocess_wav(self, path: str):
        """针对 Tacotron2 的尾部问题进行后处理"""
        y, sr = sf.read(path, dtype="float32")
        if y.size == 0:
            return

        # 统一为单声道处理
        if y.ndim > 1:
            mono = y.mean(axis=1)
        else:
            mono = y.copy()

        original_len = len(mono)
        
        # 基于能量梯度找到语音结束点
        win_ms = 30
        hop_ms = 10
        win = int(sr * win_ms / 1000)
        hop = int(sr * hop_ms / 1000)
        
        if len(mono) < win * 2:
            return
        
        n_frames = (len(mono) - win) // hop + 1
        rms = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop
            frame = mono[start:start + win]
            rms[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0
        
        if len(rms) < 3:
            return
            
        max_rms = np.max(rms)
        if max_rms == 0:
            return
        
        # 找到最后一个能量超过峰值 8% 的位置
        energy_thresh = max_rms * 0.08
        above_thresh = np.where(rms > energy_thresh)[0]
        
        if len(above_thresh) > 0:
            last_active_frame = int(np.max(above_thresh))
            energy_end = (last_active_frame * hop) + win
        else:
            energy_end = len(mono)
        
        final_end = energy_end
        
        # 确保至少保留 0.5 秒
        min_len = int(sr * 0.5)
        final_end = max(final_end, min_len)
        final_end = min(final_end, len(mono))
        
        # 应用裁切
        if final_end < len(y):
            y = y[:final_end] if y.ndim == 1 else y[:final_end]
        
        # 后处理: 淡出
        fade_len = min(int(sr * 0.05), len(y))
        if fade_len > 1:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            if y.ndim > 1:
                y[-fade_len:, :] *= fade[:, None]
            else:
                y[-fade_len:] *= fade
        
        # 防止削波
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.95:
            y = y * (0.95 / peak)
        
        sf.write(path, y, sr, subtype="PCM_16")
        
        if final_end < original_len:
            removed_ms = (original_len - final_end) * 1000 // sr
            print(f"[TTS] 裁切尾部 {removed_ms}ms")
    
    def process_tts(self, text: str) -> Optional[bytes]:
        """使用 Coqui TTS 将文本转换为语音，返回 WAV 数据"""
        if not text or not self.tts_engine:
            if not self.tts_engine:
                print("[TTS] Coqui TTS 引擎未初始化")
            return None
        
        try:
            import re
            
            # 文本预处理
            text = self._normalize_text(text)
            if not text:
                print("[TTS] 文本清洗后为空，跳过合成。")
                return None
            
            # 分割长文本
            segments = self._split_long_text(text)
            print(f"[TTS] 文本分为 {len(segments)} 段处理")
            
            all_wav_data = []
            
            for idx, segment in enumerate(segments):
                segment = segment.strip()
                if not segment:
                    continue
                    
                print(f"[TTS] 处理第 {idx + 1}/{len(segments)} 段: {segment[:50]}..." if len(segment) > 50 else f"[TTS] 处理第 {idx + 1}/{len(segments)} 段: {segment}")
                
                # 强制添加标点，帮助模型停止
                if not segment.endswith(("。", "！", "？", ".", "!", "?", "…", "~")):
                    segment += "。"
                
                # 创建临时文件
                tmp_wav = tempfile.mktemp(suffix=".wav")
                
                try:
                    # 合成
                    self.tts_engine.tts_to_file(text=segment, file_path=tmp_wav)
                    
                    # 后处理：去除尾部静音/底噪
                    try:
                        self._postprocess_wav(tmp_wav)
                    except Exception as e:
                        print(f"[TTS] 音频后处理失败 (将使用原声): {e}")
                    
                    # 读取 WAV 数据
                    if os.path.exists(tmp_wav):
                        with open(tmp_wav, "rb") as f:
                            wav_data = f.read()
                        all_wav_data.append(wav_data)
                        
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except:
                            pass
            
            if not all_wav_data:
                return None
            
            # 如果只有一段，直接返回
            if len(all_wav_data) == 1:
                print(f"[TTS] Coqui TTS 合成成功: {len(all_wav_data[0])} bytes")
                return all_wav_data[0]
            
            # 多段合并：简单拼接 PCM 数据（去掉 WAV 头）
            # 这里简化处理：返回第一段（带头），后续段的纯 PCM 需要更复杂处理
            # 为简单起见，我们返回拼接后的完整音频
            combined_wav = self._combine_wav_files(all_wav_data)
            print(f"[TTS] Coqui TTS 合成成功 (合并): {len(combined_wav)} bytes")
            return combined_wav
                    
        except Exception as e:
            print(f"[TTS] Coqui TTS 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _combine_wav_files(self, wav_data_list: list) -> bytes:
        """合并多个 WAV 数据为一个"""
        if not wav_data_list:
            return b''
        if len(wav_data_list) == 1:
            return wav_data_list[0]
        
        try:
            import io
            import wave
            
            # 读取所有音频数据
            all_frames = []
            params = None
            
            for wav_data in wav_data_list:
                with io.BytesIO(wav_data) as bio:
                    with wave.open(bio, 'rb') as wf:
                        if params is None:
                            params = wf.getparams()
                        all_frames.append(wf.readframes(wf.getnframes()))
            
            # 写入合并后的 WAV
            output = io.BytesIO()
            with wave.open(output, 'wb') as wf:
                wf.setparams(params)
                for frames in all_frames:
                    wf.writeframes(frames)
            
            return output.getvalue()
            
        except Exception as e:
            print(f"[TTS] WAV 合并失败: {e}")
            # 如果合并失败，返回第一段
            return wav_data_list[0] if wav_data_list else b''

    # ==================== 功能处理 ====================
    
    def handle_features_by_intent(self, intent: str, text: str, speaker: str = "unknown", client: Dict = None) -> Optional[str]:
        """根据意图处理功能命令"""
        
        reply = None
        
        if intent == "message_board" and self.message_board_handler:
            reply = self.message_board_handler.handle(text, speaker)
        elif intent == "schedule" and self.schedule_handler:
            reply = self.schedule_handler.handle(text)
            if reply and reply.startswith("PARTIAL_QUERY:"):
                # 处理分页逻辑
                parts = reply.split(":", 3)
                if len(parts) >= 4:
                    _, voice_text, total_str, displayed_str = parts
                    try:
                        total_count = int(total_str)
                        displayed_count = int(displayed_str)
                        if client:
                            client["dialog_state"] = "waiting_schedule_continue"
                            client["pending_schedule_data"] = {
                                "total_count": total_count,
                                "displayed_count": displayed_count,
                                "voice_text": voice_text
                            }
                        reply = voice_text
                    except ValueError:
                        pass
        elif intent == "weather" and self.weather_handler:
            reply = self.weather_handler.handle(text)
        elif intent == "news" and self.news_handler:
            reply = self.news_handler.handle(text)
        elif intent == "festival" and self.festival_handler:
            reply = self.festival_handler.handle(text)
            
        return reply

    def _handle_schedule_continue(self, text: str, client: Dict) -> Optional[str]:
        """处理日程继续念的请求"""
        text = text.lower().strip()
        
        affirmative_keywords = [
            "继续", "念完", "全部说完", "剩下的", "对", "是的", "好", "嗯",
            "yes", "yep", "go ahead", "tell me", "继续念"
        ]
        is_affirmative = any(keyword in text for keyword in affirmative_keywords)

        if not is_affirmative:
            negative_keywords = ["不用", "不念了", "算了", "no", "nope", "stop", "够了"]
            is_negative = any(keyword in text for keyword in negative_keywords)
            if is_negative:
                return "好的，不继续念了。"
            # 如果既不是肯定也不是否定，可能是在说别的，这里为了简单起见，如果不肯定就认为不念了，或者默认chat?
            # main.py 中 return None 会导致 continue loop，但这里我们需要返回回复。
            # 这里简单处理：如果不匹配肯定词，就认为是不念了，或者是新的指令？
            # 为了更好的体验，如果没匹配到，可以返回None让外层继续走意图分类。
            # 但 main.py 是如果 dialog_state 存在，就强制进入这里。
            # 修改 main.py 逻辑：如果不匹配，return None，则 dialog_state 应该保留还是清除？
            # main.py 若返回None (implicitly), dialog_state 没有被清除，下一次继续。
            # 这里我们如果返回 None, 外层会当做没有处理，进入意图分类。
            # 所以如果不是肯定也不是否定，我们返回None，并且保留 dialog_state ? 
            # 不，main.py 逻辑是:
            # if is_negative: return "..."
            # if not pending_schedule_data: return "..."
            # ...
            # 只有最后 return "继续念..."
            # 如果不匹配 affirmative 且不匹配 negative -> main.py implicitly returns None
            return None

        pending_data = client.get("pending_schedule_data", {})
        if not pending_data:
            return "抱歉，我记不清刚才的内容了。"

        total_count = pending_data.get("total_count", 0)
        displayed_count = pending_data.get("displayed_count", 0)

        remaining_items = self.schedule_handler.manager.list_items()
        # 注意：这里 list_items 可能拿到最新的，如果期间有变动可能会有微小偏差，但通常可接受
        remaining_items = remaining_items[displayed_count:]

        if not remaining_items:
            return "没有更多日程了。"

        lines = []
        for i, it in enumerate(remaining_items, displayed_count + 1):
            time_part = f"{it.time}，" if it.time else ""
            lines.append(f"第{i}条，{time_part}{it.title}，编号是{it.id}")

        return "继续念剩下的：" + " ".join(lines) + "。"

    # ==================== WebSocket 处理 ====================
    
    async def handle_client(self, websocket):
        """处理单个客户端连接"""
        client_id = id(websocket)
        client_addr = websocket.remote_address
        print(f"[Server] 客户端连接: {client_addr} (ID: {client_id})")
        
        self.clients[client_id] = {
            "websocket": websocket,
            "audio_buffer": bytearray(),
            "state": SystemState.IDLE,
            "dialog_state": None,
            "pending_schedule_data": {}
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
                
                # 异步处理 (避免阻塞 WebSocket 心跳)
                # 使用 create_task 将耗时任务扔到后台，不等待其完成
                asyncio.create_task(
                    self.process_and_respond(websocket, client_id, audio_data)
                )

    async def process_and_respond(self, websocket, client_id: str, audio_data: bytes):
        """处理音频并返回响应 (后台任务)"""
        try:
            client = self.clients.get(client_id)
            if not client:
                return

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
            
            reply_text = None
            
            # 2. 检查由多轮对话状态 (如日程继续)
            if client["dialog_state"] == "waiting_schedule_continue":
                continue_reply = await asyncio.get_event_loop().run_in_executor(
                    None, self._handle_schedule_continue, text, client
                )
                if continue_reply:
                    print(f"[Server] 日程继续回复: {continue_reply}")
                    reply_text = continue_reply
                    # 状态重置在 _handle_schedule_continue 内部或是这里做了
                    # _handle_schedule_continue 只返回文本，我们需要在这里重置状态
                    # 实际上 _handle_schedule_continue 需要访问 client pending data
                    # 我们修改了 _handle_schedule_continue 接收 client dict
                    client["dialog_state"] = None
                    client["pending_schedule_data"] = {}
            
            if not reply_text:
                # 3. 意图分类
                intent = await asyncio.get_event_loop().run_in_executor(
                    None, self.classify_intent, text
                )
                print(f"[Server] 意图分类: {intent}")
                
                # 4. 根据意图分发
                if intent == "chat":
                    reply_text = await asyncio.get_event_loop().run_in_executor(
                        None, self._generate_chat_response, text, emotion, speaker
                    )
                else:
                    # 功能处理
                    reply_text = await asyncio.get_event_loop().run_in_executor(
                        None, self.handle_features_by_intent, intent, text, speaker, client
                    )
                    if not reply_text:
                        reply_text = "抱歉，我没有理解您的意思，请再试一次。"

            print(f"[Server] 回复: {reply_text}")
            
            # 5. TTS 合成
            await self.send_state(websocket, "speaking")
            
            wav_data = await asyncio.get_event_loop().run_in_executor(
                None, self.process_tts, reply_text
            )
            
            if wav_data:
                # 发送 TTS 音频
                # 增加一个延时，确保客户端有时间处理上一条消息（如果存在并发问题）
                await asyncio.sleep(0.1)
                await websocket.send(json.dumps({
                    "type": "tts_audio",
                    "text": reply_text,
                    "data": base64.b64encode(wav_data).decode("utf-8"),
                    "is_final": True
                }))
                print("[Server] TTS音频已发送")
            
            await self.send_state(websocket, "idle")
            
        except Exception as e:
            print(f"[Server] 处理错误: {e}")
            import traceback
            traceback.print_exc()
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
