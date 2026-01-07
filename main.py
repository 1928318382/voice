# main.py - 单机模式启动入口
# 适用于在同一台机器上运行完整的语音助手（不使用客户端-服务器架构）
import warnings
import multiprocessing
import time
import sys
import os
import threading
import queue
import json
from typing import Optional

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 忽略 jieba 的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# 导入配置和核心模块
from app.core.config import SystemState
from app.core.tts import TTSEngine
from app.core.hardware import LEDController, AudioDevice
from app.core.asr import ASREngine
from app.core.llm import LLMEngine

# 导入功能模块
from app.features import (
    ScheduleCommandHandler, NewsCommandHandler, FestivalCommandHandler,
    MessageBoardCommandHandler, WeatherCommandHandler, ScheduleCategory
)


class VoiceAssistant:
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.state = SystemState.INITIALIZING

        # 硬件反馈
        self.led = LEDController(mock=mock_mode)
        # 日程管理（本地）
        self.schedule_handler = ScheduleCommandHandler()
        # 新闻查询
        self.news_handler = NewsCommandHandler()
        # 天气查询
        self.weather_handler = WeatherCommandHandler()
        # 节日提醒
        self.festival_handler = FestivalCommandHandler()
        # 留言板（稍后初始化，需要speaker_recognizer）
        self.message_board_handler = None
        self.reminder_thread = threading.Thread(target=self.reminder_loop, daemon=True)

        # 多轮对话状态管理
        self.dialog_state = None
        self.pending_schedule_data = {}

        # -----------------------------------------------------------
        # 1. 初始化情感识别引擎
        # -----------------------------------------------------------
        print("[System] 正在加载情感识别模块...")
        try:
            from app.core.emotion import EmotionRecognizer
            self.emotion_engine = EmotionRecognizer()
            self.current_emotion = "neutral"
        except Exception as e:
            print(f"[Error] 情感模块加载失败: {e}")
            self.emotion_engine = None
            self.current_emotion = "neutral"

        # -----------------------------------------------------------
        # 2. 初始化语音增强器
        # -----------------------------------------------------------
        print("[System] 正在加载语音增强模块...")
        try:
            from app.core.enhancement import AudioEnhancer
            self.audio_enhancer = AudioEnhancer()
        except Exception as e:
            print(f"[Error] 语音增强模块加载失败: {e}")
            self.audio_enhancer = None

        # -----------------------------------------------------------
        # 3. 初始化声纹识别器
        # -----------------------------------------------------------
        print("[System] 正在加载声纹识别模块...")
        try:
            from app.core.speaker import ECAPATDNNRecognizer
            self.speaker_recognizer = ECAPATDNNRecognizer()
            self.current_speaker = "unknown"
            self.message_board_handler = MessageBoardCommandHandler(self.speaker_recognizer)
        except Exception as e:
            print(f"[Error] 声纹识别模块加载失败: {e}")
            self.speaker_recognizer = None
            self.current_speaker = "unknown"
            self.message_board_handler = None

        # -----------------------------------------------------------
        # 4. 定义队列
        # -----------------------------------------------------------
        self.q_audio = multiprocessing.Queue(maxsize=2000)
        self.q_asr_output = multiprocessing.Queue()
        self.q_llm_input = multiprocessing.Queue()
        self.q_asr_cmd = multiprocessing.Queue()
        self.q_tts_text = multiprocessing.Queue()
        self.q_llm_output = multiprocessing.Queue()  # New queue for LLM output separation
        self.q_event = multiprocessing.Queue()
        self.q_cmd_input = multiprocessing.Queue()

        # -----------------------------------------------------------
        # 5. 启动子进程
        # -----------------------------------------------------------
        self.p_asr = ASREngine(
            self.q_audio, self.q_asr_output, self.q_asr_cmd,
            mock=mock_mode, enhancer=self.audio_enhancer,
            speaker_recognizer=self.speaker_recognizer
        )
        self.p_llm = LLMEngine(self.q_llm_input, self.q_llm_output, mock=mock_mode)
        self.p_tts = TTSEngine(self.q_tts_text, self.q_event, audio_device_mock=mock_mode)

        self.is_recording = False
        self.audio_buffer = bytearray()
        self._queue_overflow_count = 0

    def start(self):
        print("=" * 50)
        print("  语音交互系统 (单机模式) 启动")
        print("  [回车键]    切换 录音 / 停止并发送")
        print("  [register]  启动声纹注册工具")
        print("  [users]     查看已注册用户")
        print("  [setroot]   设置root用户 (格式: setroot 用户名)")
        print("  [q] + 回车  退出程序")
        print("=" * 50)

        self.p_asr.start()
        self.p_llm.start()
        self.p_tts.start()

        self.input_thread = threading.Thread(target=self.console_listener, daemon=True)
        self.input_thread.start()

        self.reminder_thread.start()

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
        if self.message_board_handler:
            self.message_board_handler.start_auto_cleanup()

        self._check_festival_reminders()

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
                    elif cmd == "register":
                        self.start_speaker_registration()
                    elif cmd == "users":
                        self.show_registered_users()
                    elif cmd.startswith("setroot"):
                        self.handle_setroot_command(cmd)
                    else:
                        if self.is_recording:
                            print("\n✅ 录音结束，正在分析...", end="")
                            self.is_recording = False
                            self.switch_state(SystemState.THINKING)

                            if self.emotion_engine and len(self.audio_buffer) > 0:
                                try:
                                    emo_label = self.emotion_engine.analyze(bytes(self.audio_buffer))
                                    self.current_emotion = emo_label
                                    print(f" [检测情感: {emo_label}]")
                                except Exception as e:
                                    print(f" [情感分析出错: {e}]")
                                    self.current_emotion = "neutral"
                            else:
                                self.current_emotion = "neutral"

                            self.audio_buffer.clear()
                            self.q_asr_cmd.put("COMMIT")

                        else:
                            print("\n🔴 正在录音... (说完按回车)", end="", flush=True)
                            self.is_recording = True
                            self.switch_state(SystemState.LISTENING)
                            self.audio_buffer.clear()
                            self.q_asr_cmd.put("RESET")

                # ==========================
                # 2. 读取音频硬件流
                # ==========================
                pcm = audio_dev.read_chunk()

                if self.is_recording:
                    if not self.q_audio.full():
                        self.q_audio.put(pcm)
                        self._queue_overflow_count = 0
                    else:
                        self._queue_overflow_count += 1
                        if self._queue_overflow_count % 100 == 1:
                            print(f"[Warning] 音频队列已满，丢弃数据 ({self._queue_overflow_count}帧)")

                    self.audio_buffer.extend(pcm)

                # ==========================
                # 3. 处理 ASR 识别结果并转发给 LLM
                # ==========================
                try:
                    while not self.q_asr_output.empty():
                        asr_data = self.q_asr_output.get_nowait()

                        text = ""
                        emotion = "neutral"
                        speaker = "unknown"

                        if isinstance(asr_data, dict):
                            text = asr_data.get("text", "")
                            # [MODIFIED] 优先使用主进程侦测到的实时情感，忽略 ASR 进程返回的 neutral 占位符
                            # emotion = asr_data.get("emotion", "neutral")
                            emotion = self.current_emotion if self.current_emotion else "neutral"
                            
                            speaker = asr_data.get("speaker", "unknown")
                        elif isinstance(asr_data, str):
                            text = asr_data
                            emotion = self.current_emotion if self.current_emotion else "neutral"

                        if text:
                            print(f"[Main] 识别文本: {text}")
                            if speaker != "unknown":
                                print(f"[Main] 说话人: {speaker}")
                                if self.message_board_handler:
                                    auto_notify = self.message_board_handler.notify_user_messages(speaker)
                                    if auto_notify:
                                        print(f"[Main] 自动告知留言: {auto_notify}")
                                        self.q_tts_text.put(
                                            {"text_chunk": auto_notify, "end": True}
                                        )
                                        self.current_emotion = "neutral"
                                        self.current_speaker = "unknown"
                                        continue

                            # 处理多轮对话状态（日程继续）
                            if self.dialog_state == "waiting_schedule_continue":
                                continue_reply = self._handle_schedule_continue(text)
                                if continue_reply:
                                    print(f"[Main] 日程继续回复: {continue_reply}")
                                    self.q_tts_text.put(
                                        {"text_chunk": continue_reply, "end": True}
                                    )
                                    self.dialog_state = None
                                    self.pending_schedule_data = {}
                                    self.current_emotion = "neutral"
                                    self.current_speaker = "unknown"
                                    continue

                            # 发送给LLM进行意图分类
                            packet = {
                                "text": text,
                                "emotion": emotion,
                                "speaker": speaker
                            }
                            self.q_llm_input.put(packet)

                            
                            # (Removed: "TTS reading intent in advance" logic loop as requested)
                            # We no longer wait here. The intent result will arrive in q_llm_output 
                            # and be processed by _process_llm_output() in the main loop.

                except queue.Empty:
                    pass

                # ==========================
                # 4. 处理 LLM 返回结果（意图分类或聊天回复）
                # ==========================
                self._process_llm_output()

                # ==========================
                # 5. 状态流转 (THINKING -> SPEAKING)
                # ==========================
                if not self.q_tts_text.empty() and self.state == SystemState.THINKING:
                    self.switch_state(SystemState.SPEAKING)

                # ==========================
                # 6. 监听 TTS 播放结束
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

    def _process_single_intent_result(self, llm_output):
        """处理单个意图分类结果"""
        try:
            print(f"[Main] 检测到意图分类结果: {llm_output}")
            intent_data = json.loads(llm_output["intent_result"])
            intent = intent_data.get("intent")
            text = intent_data.get("text")
            emotion = intent_data.get("emotion", "neutral")
            speaker = intent_data.get("speaker", "unknown")
            
            print(f"[Main] LLM意图分类: {intent}, 文本: {text}")
            
            # 根据意图调用对应的功能处理器
            reply = None
            if intent == "schedule":
                reply = self.schedule_handler.handle(text)
                if reply and reply.startswith("PARTIAL_QUERY:"):
                    parts = reply.split(":", 3)
                    if len(parts) >= 4:
                        _, voice_text, total_str, displayed_str = parts
                        try:
                            total_count = int(total_str)
                            displayed_count = int(displayed_str)
                            self.dialog_state = "waiting_schedule_continue"
                            self.pending_schedule_data = {
                                "total_count": total_count,
                                "displayed_count": displayed_count,
                                "voice_text": voice_text
                            }
                            reply = voice_text
                        except ValueError:
                            pass
            elif intent == "weather":
                print(f"[Main] 调用天气处理器处理: {text}")
                reply = self.weather_handler.handle(text)
                print(f"[Main] 天气处理器返回: {reply}")
            elif intent == "news":
                print(f"[Main] 调用新闻处理器处理: {text}")
                reply = self.news_handler.handle(text)
                print(f"[Main] 新闻处理器返回: {reply}")
            elif intent == "festival":
                reply = self.festival_handler.handle(text)
            elif intent == "message_board":
                reply = self.message_board_handler.handle(text, speaker) if self.message_board_handler else None
            
            if reply:
                print(f"[Main] 功能处理器回复: {reply}")
                self.q_tts_text.put({"text_chunk": reply, "end": True})
            else:
                # 如果功能处理器没有返回结果，返回默认回复
                print(f"[Main] 功能处理器无回复，返回默认提示")
                default_reply = "抱歉，我没有理解您的意思，请再试一次。"
                self.q_tts_text.put({"text_chunk": default_reply, "end": True})
        except Exception as e:
            print(f"[Main] 处理意图结果时出错: {e}")
            import traceback
            traceback.print_exc()

    def _process_llm_output(self):
        """处理LLM返回结果（意图分类或聊天回复），从 q_llm_output 读取"""
        try:
            # 检查队列中是否有数据
            while not self.q_llm_output.empty():
                try:
                    llm_output = self.q_llm_output.get_nowait()
                    
                    # 检查是否是意图分类结果
                    if "intent_result" in llm_output:
                        self._process_single_intent_result(llm_output)
                    else:
                        # 正常的聊天回复chunk，直接转发给TTS
                        # We forward it to TTS immediately, no need for buffering or putting back
                        self.q_tts_text.put(llm_output)

                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[Main] 处理单个LLM输出项时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                        
        except Exception as e:
            print(f"[Main] 处理LLM输出时出错: {e}")
            import traceback
            traceback.print_exc()

    def _build_reminder_text(self, item):
        """生成提醒话术"""
        reminder_text = getattr(item, "reminder_text", None)
        if reminder_text:
            return reminder_text

        title = item.title or "该做的事情"
        has_time = bool(item.time)

        if item.category == ScheduleCategory.MEDICATION:
            if has_time:
                return f"现在到吃药时间啦，记得{title}。"
            return f"别忘了吃药哦，记得{title}。"

        if item.category == ScheduleCategory.ROUTINE:
            if has_time:
                return f"现在到时间啦，按计划该{title}了。"
            return f"提醒你按作息安排，记得{title}。"

        if item.category == ScheduleCategory.TODO:
            if has_time:
                return f"现在差不多该处理一下待办啦，记得{title}。"
            return f"提醒你有个事情别忘了，记得{title}。"

        if has_time:
            return f"现在到你设定的时间啦，记得{title}。"
        return f"提醒你一下，记得{title}。"

    def reminder_loop(self):
        """后台轮询日程，按时间主动提醒"""
        from datetime import datetime

        while True:
            try:
                now = datetime.now()
                manager = self.schedule_handler.manager
                due_items = manager.get_due_items(now)
                for it in due_items:
                    msg = self._build_reminder_text(it)
                    print(f"[Reminder] {msg}")
                    self.q_tts_text.put({"text_chunk": msg, "end": True})

                    if it.time and len(it.time) == 16 and " " in it.time:
                        try:
                            manager.delete_item(it.id)
                            print(f"[Reminder] 已自动删除一次性日程（编号 {it.id}）")
                        except Exception as e:
                            print(f"[Reminder] 自动删除日程失败: {e}")
            except Exception as e:
                print(f"[Reminder] 定时提醒出错: {e}")

            time.sleep(30)

    def start_speaker_registration(self):
        """启动声纹注册流程"""
        print("\n🎤 启动声纹注册工具...")
        try:
            from tools.register_speaker import SpeakerRegistrationTool

            tool = SpeakerRegistrationTool()
            tool.run()

            print("\n✅ 返回语音助手主界面")
            print("按回车键继续对话...")

        except Exception as e:
            print(f"❌ 启动注册工具失败: {e}")
            print("请手动运行: python tools/register_speaker.py")

    def show_registered_users(self):
        """显示已注册用户"""
        try:
            users = self.speaker_recognizer.get_user_list()
            if users:
                root_user = self.speaker_recognizer.get_root_user()
                print(f"\n👥 已注册用户 ({len(users)} 个):")
                for user in users:
                    count = self.speaker_recognizer.get_user_count(user)
                    is_root = "👑" if user == root_user else ""
                    status = "✅" if count >= 3 else "⚠️ "
                    print(f"  {status} {is_root} {user}: {count} 个样本")
                if root_user:
                    print(f"\n👑 Root用户: {root_user}")
            else:
                print("\n📭 暂无注册用户")
                print("输入 'register' 开始注册声纹")
        except Exception as e:
            print(f"❌ 获取用户列表失败: {e}")

    def handle_setroot_command(self, cmd: str):
        """处理设置root用户命令"""
        try:
            parts = cmd.split()
            if len(parts) < 2:
                print("\n❌ 用法: setroot 用户名")
                print("例如: setroot user001")
                return

            user_id = parts[1]
            if self.speaker_recognizer:
                success = self.speaker_recognizer.set_root_user(user_id)
                if success:
                    print(f"\n✅ 已设置 {user_id} 为root用户")
                else:
                    print(f"\n❌ 设置失败：用户 {user_id} 未注册")
            else:
                print("\n❌ 声纹识别器未初始化")
        except Exception as e:
            print(f"\n❌ 设置root用户失败: {e}")

    def _handle_schedule_continue(self, text: str) -> Optional[str]:
        """处理多轮对话：用户确认是否继续念剩下的日程"""
        text = text.lower().strip()

        affirmative_keywords = [
            "继续", "念完", "全部说完", "剩下的", "对", "是的", "好", "嗯",
            "yes", "yep", "go ahead", "tell me", "继续念"
        ]

        is_affirmative = any(keyword in text for keyword in affirmative_keywords)

        if not is_affirmative:
            negative_keywords = [
                "不用", "不念了", "算了", "no", "nope", "stop", "够了"
            ]
            is_negative = any(keyword in text for keyword in negative_keywords)
            if is_negative:
                return "好的，不继续念了。"

        if not self.pending_schedule_data:
            return "抱歉，我记不清刚才的内容了。"

        total_count = self.pending_schedule_data.get("total_count", 0)
        displayed_count = self.pending_schedule_data.get("displayed_count", 0)

        remaining_items = self.schedule_handler.manager.list_items()
        remaining_items = remaining_items[displayed_count:]

        if not remaining_items:
            return "没有更多日程了。"

        lines = []
        for i, it in enumerate(remaining_items, displayed_count + 1):
            time_part = f"{it.time}，" if it.time else ""
            lines.append(f"第{i}条，{time_part}{it.title}，编号是{it.id}")

        return "继续念剩下的：" + " ".join(lines) + "。"

    def _check_festival_reminders(self):
        """检查并播放节日提醒"""
        try:
            festival_reminder = self.festival_handler.check_and_remind_festivals()
            if festival_reminder:
                print(f"[Festival] 节日提醒: {festival_reminder}")
                self.q_tts_text.put({"text_chunk": festival_reminder, "end": True})
                time.sleep(2)
        except Exception as e:
            print(f"[Festival] 检查节日提醒失败: {e}")

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
