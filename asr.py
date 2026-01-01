import multiprocessing
import queue
import time
import os
import numpy as np
import tempfile
import uuid
import soundfile as sf
import shutil

# 引入我们在 config.py 中新定义的绝对路径
from config import ASR_MODEL_PATH, SAMPLE_RATE, BASE_DIR
from enhancement import AudioEnhancer
# 注意：SpeakerRecognizer 和 EmotionRecognizer 已移至主进程，避免子进程重复加载
from funasr import AutoModel


class ASREngine(multiprocessing.Process):
    def __init__(self, audio_queue, text_queue, command_queue, mock=False):
        super().__init__()
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.command_queue = command_queue
        self.mock = mock
        self.running = True
        self.audio_buffer = []

    def run(self):
        print("[ASR] 进程启动...")

        # --- 初始化辅助模块 ---
        # 语音增强模块（降噪等）
        enhancer = AudioEnhancer()
        # 注意：声纹识别和情感识别已移至主进程处理

        model_instance = None

        # --- 加载 ASR 模型逻辑 ---
        if not self.mock:
            # 检查下载标记文件
            flag_file = os.path.join(BASE_DIR, ".asr_download_ok")

            if not os.path.exists(flag_file):
                print("[ASR] ⚠️ 未检测到模型下载标记 (.asr_download_ok)")
                print("[ASR] 请先运行 python download.py 下载模型！")
                print("[ASR] -> 自动切换为 Mock 模式")
                self.mock = True
            elif not os.path.exists(ASR_MODEL_PATH):
                print(f"[ASR] ⚠️ 未找到模型文件夹: {ASR_MODEL_PATH}")
                print("[ASR] -> 自动切换为 Mock 模式")
                self.mock = True
            else:
                try:
                    print(f"[ASR] 正在加载本地模型: {ASR_MODEL_PATH} ...")

                    # 【关键修改】使用本地绝对路径加载模型
                    # 这样 FunASR 就不会去联网查找，而是直接读取该文件夹
                    model_instance = AutoModel(
                        model=ASR_MODEL_PATH,
                        device="cpu",  # 树莓派强制 CPU
                        disable_update=True,  # 禁止检查更新
                        disable_log=True,
                        trust_remote_code=True
                    )
                    print("[ASR] ✅ 模型加载成功 (Offline Mode)")

                except Exception as e:
                    print(f"[ASR] ❌ 模型加载失败: {e}")
                    print("[ASR] -> 自动切换为 Mock 模式")
                    self.mock = True

        # --- 主循环 ---
        while self.running:
            try:
                # 1. 处理控制指令 (非阻塞)
                while not self.command_queue.empty():
                    try:
                        cmd = self.command_queue.get(block=False)
                        if cmd == "STOP":
                            self.running = False
                            break
                        elif cmd == "RESET":
                            self.audio_buffer = []
                        elif cmd == "COMMIT":
                            if self.audio_buffer:
                                self.process_buffer(model_instance)
                            else:
                                print("[ASR] 缓冲区为空，忽略 COMMIT")
                    except queue.Empty:
                        pass

                if not self.running:
                    break

                # 2. 获取音频数据 (带超时，避免死循环占用 CPU)
                try:
                    chunk = self.audio_queue.get(timeout=0.05)
                    # 可选：在此处调用 enhancer.process(chunk)
                    self.audio_buffer.append(chunk)
                except queue.Empty:
                    continue

            except Exception as e:
                print(f"[ASR] 运行循环异常: {e}")
                time.sleep(1)  # 避免报错刷屏

    def process_buffer(self, model):
        """将缓冲区音频拼接并送入模型识别"""
        # 简单检查缓冲区大小，避免太短的误触（提升至10帧=0.6秒，更可靠）
        if len(self.audio_buffer) < 10:
            print("[ASR] 音频过短（<0.6秒），忽略")
            self.audio_buffer = []
            return

        print(f"[ASR] 开始识别 ({len(self.audio_buffer)} 帧)...")
        text = ""
        emotion_tag = "neutral"

        if self.mock or model is None:
            # Mock 模式：模拟延迟并返回测试文本
            time.sleep(0.5)
            text = "你好，这是树莓派离线测试"
        else:
            # 1. 拼接音频
            full_audio = b"".join(self.audio_buffer)

            # 2. 写入临时文件 (FunASR 目前对文件输入支持最稳)
            tmp_file = os.path.join(tempfile.gettempdir(), f"speech_{uuid.uuid4()}.wav")

            try:
                # 转换为 float32 并保存 wav (16k mono)
                # 假设输入是 int16 PCM
                audio_np = np.frombuffer(full_audio, dtype=np.int16)

                # 安全检查：避免空数据
                if len(audio_np) == 0:
                    return

                # 写入 WAV 文件
                sf.write(tmp_file, audio_np, SAMPLE_RATE)

                # 3. 推理
                # batch_size_s 设大一点避免切分，树莓派上通常处理单句指令
                res = model.generate(input=tmp_file, batch_size_s=300, disable_pbar=True)

                # 4. 解析结果
                # res 通常是List: [{'key': '...', 'text': '识别结果'}]
                if isinstance(res, list) and len(res) > 0:
                    text = res[0].get("text", "")
                elif isinstance(res, dict):
                    text = res.get("text", "")

            except Exception as e:
                print(f"[ASR] 推理过程出错: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass

        # --- 后处理与输出 ---
        if text:
            # 清理标点或空格
            text = text.replace(" ", "").strip()
            print(f"[ASR] 识别结果: 【{text}】")

            # 发送给主进程 (Main -> LLM)
            self.text_queue.put({"text": text, "emotion": emotion_tag})
        else:
            print("[ASR] 未识别到有效内容")

        # 清空缓冲区，准备下一次
        self.audio_buffer = []