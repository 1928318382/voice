# tts.py
import os
import sys
import time
import queue
import tempfile
import subprocess
import multiprocessing
import numpy as np
import re

# 引入音频处理库
import soundfile as sf

# 在导入 torch / TTS 之前关闭 weights_only 限制（适配 torch>=2.6）
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from TTS.api import TTS as CoquiTTS  # 来自 coqui-tts
from app.core.config import TTS_MODEL_PATH, TTS_CONFIG_PATH

# ================= 数字转中文 =================

CHINESE_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
CHINESE_UNITS = ['', '十', '百', '千']
CHINESE_BIG_UNITS = ['', '万', '亿', '兆']

def _number_to_chinese(num_str: str) -> str:
    """将数字字符串转换为中文读法"""
    if not num_str:
        return ""
    
    # 处理小数
    if '.' in num_str:
        parts = num_str.split('.', 1)
        integer_part = _integer_to_chinese(parts[0])
        decimal_part = ''.join(CHINESE_DIGITS[int(d)] for d in parts[1] if d.isdigit())
        return f"{integer_part}点{decimal_part}" if decimal_part else integer_part
    
    return _integer_to_chinese(num_str)

def _integer_to_chinese(num_str: str) -> str:
    """将整数字符串转换为中文"""
    if not num_str:
        return ""
    
    # 去除前导零
    num_str = num_str.lstrip('0') or '0'
    
    if num_str == '0':
        return '零'
    
    # 处理负数
    if num_str.startswith('-'):
        return '负' + _integer_to_chinese(num_str[1:])
    
    # 对于特别长的数字（如电话号码），逐位读出
    if len(num_str) > 8:
        return ''.join(CHINESE_DIGITS[int(d)] for d in num_str if d.isdigit())
    
    result = []
    length = len(num_str)
    
    # 按4位一组处理
    groups = []
    while num_str:
        groups.append(num_str[-4:])
        num_str = num_str[:-4]
    groups.reverse()
    
    for group_idx, group in enumerate(groups):
        group_result = _four_digits_to_chinese(group)
        if group_result:
            big_unit_idx = len(groups) - group_idx - 1
            if big_unit_idx < len(CHINESE_BIG_UNITS):
                result.append(group_result + CHINESE_BIG_UNITS[big_unit_idx])
            else:
                result.append(group_result)
    
    return ''.join(result)

def _four_digits_to_chinese(num_str: str) -> str:
    """将4位以内的数字转换为中文"""
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

def _convert_dates_in_text(text: str) -> str:
    """
    将日期格式 (如 2026-1-7, 2026/01/07) 转换为中文读法
    例如: 2026-1-7 -> 二零二六年一月七日
    """
    def replace_date(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        
        # 年份逐位读
        year_chinese = "".join(CHINESE_DIGITS[int(d)] for d in year)
        
        # 月份和日期按整数读
        month_chinese = _integer_to_chinese(month)
        day_chinese = _integer_to_chinese(day)
        
        return f"{year_chinese}年{month_chinese}月{day_chinese}日"

    # 匹配 YYYY-MM-DD 或 YYYY/MM/DD
    # 允许月/日有一位或两位
    pattern = r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:日)?'
    return re.sub(pattern, replace_date, text)

def _convert_numbers_in_text(text: str) -> str:
    """将文本中的数字转换为中文读法"""
    def replace_number(match):
        num = match.group(0)
        return _number_to_chinese(num)
    
    # 匹配整数和小数
    # (?<!\d) 确保负号前不是数字（避免把 5-10 识别为 5 和 -10）
    pattern = r'(?<!\d)-?\d+\.?\d*'
    return re.sub(pattern, replace_number, text)

# 句子最大长度（字符数），超过则分段
MAX_SENTENCE_LENGTH = 80


class TTSEngine(multiprocessing.Process):
    """
    TTS 进程（仅使用本地 Coqui TTS 模型）
    """

    def __init__(self, input_queue, event_queue, audio_device_mock: bool = False):
        super().__init__()
        self.input_queue = input_queue
        self.event_queue = event_queue
        self.mock = audio_device_mock

        self.text_buffer = ""
        self._tts = None  # Coqui TTS 实例

    # ================= 进程主循环 =================

    def run(self):
        print("[TTS] 进程启动（后端：本地 Coqui TTS）")
        # 尝试预初始化一次
        self._ensure_tts()

        if self.mock:
            print("[TTS] mock 模式：只打印文本，不实际播放。")

        while True:
            try:
                data = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, KeyboardInterrupt):
                break

            try:
                chunk = data.get("text_chunk", "")
                is_end = data.get("end", False)
            except Exception:
                continue

            if chunk:
                self.text_buffer += chunk

            if is_end:
                text_to_speak = self.text_buffer.strip()
                self.text_buffer = ""

                if text_to_speak:
                    print(f"[TTS] 合成并播放: {text_to_speak}")
                    if not self.mock:
                        self._speak(text_to_speak)
                    else:
                        print("[TTS][mock]", text_to_speak)

                # 通知主进程这一轮 TTS 已结束
                try:
                    self.event_queue.put("TTS_FINISHED")
                except Exception:
                    pass

    # ================= Coqui TTS 初始化 =================

    def _ensure_tts(self):
        """懒加载 Coqui TTS 模型"""
        if self._tts is not None:
            return self._tts

        try:
            print("[TTS] 正在初始化本地 Coqui TTS ...")
            if not (os.path.exists(TTS_MODEL_PATH) and os.path.exists(TTS_CONFIG_PATH)):
                raise FileNotFoundError(f"模型文件未找到: {TTS_MODEL_PATH}")

            self._tts = CoquiTTS(
                model_path=TTS_MODEL_PATH,
                config_path=TTS_CONFIG_PATH,
                progress_bar=False,
                gpu=False,
            )
            print("[TTS] 本地 Coqui TTS 初始化成功。")
        except Exception as e:
            print("[TTS] 初始化本地 Coqui 失败：", repr(e))
            self._tts = None

        return self._tts

    # ================= 播放入口 =================

    def _speak(self, text: str):
        tts = self._ensure_tts()
        if tts is None:
            print("[TTS] 无可用 Coqui 引擎，跳过朗读。")
            return

        try:
            text = self._normalize_text(text)
            if not text:
                print("[TTS] 文本清洗后为空，跳过朗读。")
                return
            
            # [NEW] 检查是否包含英文字符
            has_english = bool(re.search(r'[A-Za-z]', text))
            
            use_fallback = False
            if has_english and sys.platform == "darwin":
                print("[TTS] 检测到英文，使用系统 TTS (macOS say) 以确保发音准确。")
                use_fallback = True

            if not use_fallback:
                try:
                    self._speak_coqui(tts, text)
                except Exception as e:
                    print(f"[TTS] Coqui 合成失败 ({e})，尝试使用系统 TTS 回退...")
                    use_fallback = True
            
            if use_fallback:
                self._speak_system(text)

        except Exception as e:
            print(f"[TTS] 合成/播放失败: {e}")

    def _speak_system(self, text: str):
        """使用系统自带 TTS (macOS say)"""
        if self.mock:
            print(f"[TTS][mock] 系统TTS: {text}")
            return

        try:
            if sys.platform == "darwin":
                # 使用 macOS 的 say 命令，指定中文语音 (Ting-Ting 或 Sin-Ji)，如果包含英文它会自动处理
                subprocess.run(["say", text], check=False)
            else:
                print("[TTS] 当前系统不支持 fallback TTS")
        except Exception as e:
            print(f"[TTS] 系统 TTS 失败: {e}")

    # ================= Coqui: 合成 -> 切除尾音 -> 播放 =================

    def _split_long_text(self, text: str) -> list:
        """
        将长文本按句子边界分割，确保每段不超过 MAX_SENTENCE_LENGTH
        """
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
            if len(sentence) > MAX_SENTENCE_LENGTH:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                # 按逗号拆分
                sub_sentences = re.split(r'([，,、])', sentence)
                sub_buffer = ""
                for sub in sub_sentences:
                    if len(sub_buffer) + len(sub) <= MAX_SENTENCE_LENGTH:
                        sub_buffer += sub
                    else:
                        if sub_buffer:
                            result.append(sub_buffer)
                        sub_buffer = sub
                if sub_buffer:
                    result.append(sub_buffer)
            elif len(buffer) + len(sentence) <= MAX_SENTENCE_LENGTH:
                buffer += sentence
            else:
                if buffer:
                    result.append(buffer)
                buffer = sentence
        
        if buffer:
            result.append(buffer)
        
        return result if result else [text]

    def _speak_coqui(self, tts: CoquiTTS, text: str):
        """
        1. 长文本分段处理
        2. 文本预处理（加句号）
        3. 合成 wav
        4. 轻量后处理：削减尾部静音/杂音，避免削波失真
        5. 播放
        """
        
        # 分割长文本
        segments = self._split_long_text(text)
        print(f"[TTS] 文本分为 {len(segments)} 段处理")
        
        for idx, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
                
            print(f"[TTS] 处理第 {idx + 1}/{len(segments)} 段: {segment[:50]}..." if len(segment) > 50 else f"[TTS] 处理第 {idx + 1}/{len(segments)} 段: {segment}")
            
            # 强制添加标点，帮助模型停止
            if not segment.endswith(("。", "！", "？", ".", "!", "?", "…", "~")):
                segment += "。"

            # 创建临时文件 (Windows 安全写法：先 close 再用)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.close()
                tmp_wav = f.name

            try:
                # 合成
                tts.tts_to_file(text=segment, file_path=tmp_wav)

                # 后处理：去除尾部静音/底噪
                try:
                    self._postprocess_wav(tmp_wav)
                except Exception as e:
                    print(f"[TTS] 音频后处理失败 (将播放原声): {e}")

                # 播放
                self._play_wav(tmp_wav)

            finally:
                # 清理文件
                if os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except OSError:
                        pass

    def _play_wav(self, path: str):
        if self.mock:
            print(f"[TTS][mock] 模拟播放：{path}")
            return

        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
            elif sys.platform == "darwin":
                subprocess.run(["afplay", path], check=False)
            else:
                subprocess.run(["aplay", path], check=False)
        except Exception as e:
            print(f"[TTS] 播放 wav 失败: {e}")

    def _normalize_text(self, text: str) -> str:
        """清洗文本以减少 TTS 词表缺失导致的异常发音。"""
        if not text:
            return ""
        
        # [NEW] Convert newlines to sentence delimiters to preserve list structure
        text = re.sub(r'\n+', '。', text)
        
        # 先处理日期 [NEW]
        text = _convert_dates_in_text(text)

        # 先将数字转换为中文读法
        text = _convert_numbers_in_text(text)
        
        # 去掉英文/拼音（保留中文和转换后的数字）
        # [MODIFIED] 保留英文，以便支持中英混合或纯英文 (如果有 fallback)
        # text = re.sub(r"[A-Za-z]+", " ", text)
        
        # 清理不常见符号 (保留字母)
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z，。！？、；：,.!?…~\s]", " ", text)
        # 统一空白
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ================= 轻量后处理 =================

    def _postprocess_wav(self, path: str):
        """
        针对 Tacotron2 的尾部问题进行激进后处理：
        1. 检测并移除重复模式（循环怪声）
        2. 检测能量骤降点作为结束位置
        3. 应用淡出避免爆音
        """
        y, sr = sf.read(path, dtype="float32")
        if y.size == 0:
            return

        # 统一为单声道处理
        if y.ndim > 1:
            mono = y.mean(axis=1)
        else:
            mono = y.copy()

        original_len = len(mono)
        
        # ===== 策略1: 基于能量梯度找到语音结束点 =====
        # 使用更短的窗口以便精确定位
        win_ms = 30  # 30ms 窗口
        hop_ms = 10  # 10ms 步进
        win = int(sr * win_ms / 1000)
        hop = int(sr * hop_ms / 1000)
        
        if len(mono) < win * 2:
            return  # 太短，不处理
        
        n_frames = (len(mono) - win) // hop + 1
        rms = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop
            frame = mono[start:start + win]
            rms[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0
        
        if len(rms) < 3:
            return
            
        # 找到 RMS 峰值位置
        max_rms = np.max(rms)
        if max_rms == 0:
            return
        
        # ===== 策略2: 检测重复模式（自相关法）=====
        # Tacotron2 尾部重复通常有固定周期
        def detect_repetition_start(signal, sr, min_period_ms=50, max_period_ms=300):
            """检测重复模式开始的位置"""
            min_period = int(sr * min_period_ms / 1000)
            max_period = int(sr * max_period_ms / 1000)
            
            # 只分析后30%部分（更保守）
            analyze_start = int(len(signal) * 0.7)
            segment = signal[analyze_start:]
            
            if len(segment) < max_period * 3:
                return None
            
            # 计算短时自相关寻找重复
            chunk_size = int(sr * 0.1)  # 100ms 块
            n_chunks = len(segment) // chunk_size
            
            for i in range(n_chunks - 2):
                chunk1 = segment[i * chunk_size:(i + 1) * chunk_size]
                chunk2 = segment[(i + 1) * chunk_size:(i + 2) * chunk_size]
                
                # 计算归一化互相关
                if np.std(chunk1) > 0.001 and np.std(chunk2) > 0.001:
                    corr = np.corrcoef(chunk1, chunk2)[0, 1]
                    if corr > 0.92:  # 更高阈值 = 更保守，只切真正重复的
                        return analyze_start + i * chunk_size
            
            return None
        
        repetition_start = detect_repetition_start(mono, sr)
        
        # ===== 策略3: 基于能量下降找结束点 =====
        # 找到最后一个能量超过峰值 8% 的位置（更保守）
        energy_thresh = max_rms * 0.08
        above_thresh = np.where(rms > energy_thresh)[0]
        
        if len(above_thresh) > 0:
            last_active_frame = int(np.max(above_thresh))
            energy_end = (last_active_frame * hop) + win
        else:
            energy_end = len(mono)
        
        # ===== 综合决策 =====
        # 取最早的结束点
        cut_points = [energy_end]
        if repetition_start is not None:
            cut_points.append(repetition_start)
        
        final_end = min(cut_points)
        
        # 确保至少保留 0.5 秒
        min_len = int(sr * 0.5)
        final_end = max(final_end, min_len)
        final_end = min(final_end, len(mono))
        
        # 应用裁切
        if final_end < len(y):
            y = y[:final_end] if y.ndim == 1 else y[:final_end]
            mono = mono[:final_end]
        
        # ===== 后处理: 淡出 =====
        fade_len = min(int(sr * 0.05), len(y))  # 50ms 淡出
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
            print(f"[TTS] 裁切尾部 {removed_ms}ms (重复/噪声)")


if __name__ == "__main__":
    # 单文件测试
    q_in = multiprocessing.Queue()
    q_evt = multiprocessing.Queue()
    tts_proc = TTSEngine(q_in, q_evt, audio_device_mock=False)
    tts_proc.start()

    q_in.put({"text_chunk": "你好，这是一段测试音频。", "end": True})
    
    try:
        evt = q_evt.get(timeout=15)
        print("[TEST] 收到事件:", evt)
    except queue.Empty:
        print("[TEST] 超时")

    tts_proc.terminate()
