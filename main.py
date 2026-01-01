# main.py
import multiprocessing
import time
import sys
import threading
import queue  # 导入 queue 模块用于处理 Empty 异常

from config import SystemState
from tts import TTSEngine

# 导入功能模块 (Emotion 模块改为在类中懒加载)
from hardware import LEDController, AudioDevice
from asr import ASREngine
from llm import LLMEngine

class VoiceAssistant:
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.state = SystemState.INITIALIZING

        # 硬件反馈
        self.led = LEDController(mock=mock_mode)

        # -----------------------------------------------------------
        # 1. 初始化情感识别引擎
        #    (改为在方法内部导入，防止子进程启动时重复加载导致崩溃)
        # -----------------------------------------------------------
        print("[System] 正在加载情感识别模块...")
        try:
            from emotion import EmotionRecognizer  # <--- 关键修改：懒加载
            self.emotion_engine = EmotionRecognizer()
            self.current_emotion = "neutral"  # 默认情感
        except Exception as e:
            print(f"[Error] 情感模块加载失败: {e}")
            self.emotion_engine = None
            self.current_emotion = "neutral"

        # -----------------------------------------------------------
        # 2. 定义队列
        # -----------------------------------------------------------
        self.q_audio = multiprocessing.Queue(maxsize=2000)      # Mic -> ASR (原始PCM)
        
        # 将 ASR 和 LLM 的连接断开，中间由主线程中转
        self.q_asr_output = multiprocessing.Queue()             # ASR -> Main (识别结果)
        self.q_llm_input = multiprocessing.Queue()              # Main -> LLM (文本+情感)
        
        self.q_asr_cmd = multiprocessing.Queue()                # Main -> ASR (控制指令)
        self.q_tts_text = multiprocessing.Queue()               # LLM -> TTS (流式文本)
        self.q_event = multiprocessing.Queue()                  # TTS -> Main (播放结束事件)
        self.q_cmd_input = multiprocessing.Queue()              # Keyboard -> Main

        # -----------------------------------------------------------
        # 3. 启动子进程
        # -----------------------------------------------------------
        # ASR 进程：输出到 q_asr_output
        self.p_asr = ASREngine(self.q_audio, self.q_asr_output, self.q_asr_cmd, mock=mock_mode)
        
        # LLM 进程：输入改为 q_llm_input
        self.p_llm = LLMEngine(self.q_llm_input, self.q_tts_text, mock=mock_mode)
        
        self.p_tts = TTSEngine(self.q_tts_text, self.q_event, audio_device_mock=mock_mode)

        self.is_recording = False
        
        # 音频缓冲区 (用于情感分析)
        self.audio_buffer = bytearray()
        
        # 队列溢出计数器（避免日志刷屏）
        self._queue_overflow_count = 0

    def start(self):
        print("=" * 50)
        print("  语音交互系统 (含情感识别) 启动")
        print("  [回车键]  切换 录音 / 停止并发送")
        print("  [q] + 回车  退出程序")
        print("=" * 50)

        self.p_asr.start()
        self.p_llm.start()
        self.p_tts.start()

        self.input_thread = threading.Thread(target=self.console_listener, daemon=True)
        self.input_thread.start()

        self.switch_state(SystemState.IDLE)
        self.run_loop()

    def console_listener(self):
        """后台线程监听键盘输入"""
        while True:
            try:
                cmd = input()
                self.q_cmd_input.put(cmd.strip().lower())
            except EOFError:
                break

    def run_loop(self):
        audio_dev = AudioDevice(mock=self.mock_mode)
        audio_dev.start_stream()
        print("\n[System] 就绪。按回车开始对话...")

        try:
            while True:
                # ==========================
                # 1. 处理键盘交互
                # ==========================
                if not self.q_cmd_input.empty():
                    cmd = self.q_cmd_input.get()

                    if cmd == "q":
                        self.shutdown()
                    else:
                        if self.is_recording:
                            # -------- 停止录音 --------
                            print("\n✅ 录音结束，正在分析...", end="")
                            self.is_recording = False
                            self.switch_state(SystemState.THINKING)
                            
                            # A. 执行情感分析 (使用 buffer 中的数据)
                            if self.emotion_engine and len(self.audio_buffer) > 0:
                                try:
                                    # 注意：emotion_engine.analyze 需要 bytes 类型
                                    emo_label = self.emotion_engine.analyze(bytes(self.audio_buffer))
                                    self.current_emotion = emo_label
                                    print(f" [检测情感: {emo_label}]")
                                except Exception as e:
                                    print(f" [情感分析出错: {e}]")
                                    self.current_emotion = "neutral"
                            else:
                                self.current_emotion = "neutral"
                            
                            # 清空音频缓冲，准备下一次
                            self.audio_buffer.clear()

                            # B. 通知 ASR 提交识别
                            self.q_asr_cmd.put("COMMIT")

                        else:
                            # -------- 开始录音 --------
                            print("\n🔴 正在录音... (说完按回车)", end="", flush=True)
                            self.is_recording = True
                            self.switch_state(SystemState.LISTENING)
                            self.audio_buffer.clear() # 确保缓冲干净
                            self.q_asr_cmd.put("RESET")

                # ==========================
                # 2. 读取音频硬件流
                # ==========================
                pcm = audio_dev.read_chunk()

                if self.is_recording:
                    # 分发音频数据
                    # 1. 给 ASR (用于转文字)
                    if not self.q_audio.full():
                        self.q_audio.put(pcm)
                        self._queue_overflow_count = 0  # 重置计数器
                    else:
                        self._queue_overflow_count += 1
                        # 每100次溢出只提示一次，避免刷屏
                        if self._queue_overflow_count % 100 == 1:
                            print(f"[Warning] 音频队列已满，丢弃数据 ({self._queue_overflow_count}帧)")
                    
                    # 2. 给 Emotion (存入缓冲)
                    self.audio_buffer.extend(pcm)

                # ==========================
                # 3. 处理 ASR 识别结果并转发给 LLM
                # ==========================
                try:
                    # 检查是否有 ASR 结果输出
                    while not self.q_asr_output.empty():
                        asr_data = self.q_asr_output.get_nowait()
                        
                        # 兼容处理：asr_data 可能是纯文本字符串，也可能是字典
                        text = ""
                        if isinstance(asr_data, dict):
                            text = asr_data.get("text", "")
                        elif isinstance(asr_data, str):
                            text = asr_data
                        
                        if text:
                            print(f"[Main] 识别文本: {text}")
                            # --- 关键步骤：打包 文本 + 情感 发给 LLM ---
                            packet = {
                                "text": text,
                                "emotion": self.current_emotion  # 附带刚才识别的情感
                            }
                            self.q_llm_input.put(packet)
                            
                            # 重置情感状态 (防止影响下一句)
                            self.current_emotion = "neutral"

                except queue.Empty:
                    pass

                # ==========================
                # 4. 状态流转 (THINKING -> SPEAKING)
                # ==========================
                if not self.q_tts_text.empty() and self.state == SystemState.THINKING:
                    self.switch_state(SystemState.SPEAKING)

                # ==========================
                # 5. 监听 TTS 播放结束
                # ==========================
                while not self.q_event.empty():
                    evt = self.q_event.get()
                    if evt == "TTS_FINISHED" and not self.is_recording:
                        self.switch_state(SystemState.IDLE)
                        print("\n[System] 回复完毕。按回车继续...")

                time.sleep(0.002)

        except KeyboardInterrupt:
            self.shutdown()

    def switch_state(self, s: SystemState):
        self.state = s
        self.led.set_state(s)

    def shutdown(self):
        print("\n正在退出...")
        self.p_asr.terminate()
        self.p_llm.terminate()
        self.p_tts.terminate()
        sys.exit(0)


if __name__ == "__main__":
    # Windows下多进程必须放在 if __name__ == "__main__": 之下
    app = VoiceAssistant(mock_mode=False) 
    app.start()