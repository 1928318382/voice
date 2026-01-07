import multiprocessing
import queue
import time
import json
from app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class LLMEngine(multiprocessing.Process):
    """
    认知引擎进程 (API版)
    输入: 用户文本 (Queue)
    输出: 助手回复的流式文本 (Queue) 或 意图分类结果
    """

    def __init__(self, input_queue, output_queue, mock=False):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.mock = mock

    def classify_intent(self, user_text: str, client) -> str:
        """
        使用LLM对用户输入进行意图分类
        返回: schedule, weather, news, festival, message_board, chat
        """
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
            import signal
            
            # 设置超时（5秒）
            def timeout_handler(signum, frame):
                raise TimeoutError("意图分类超时")
            
            # Windows不支持signal.alarm，使用threading.Timer代替
            import threading
            timeout_occurred = threading.Event()
            
            def timeout_callback():
                timeout_occurred.set()
            
            timer = threading.Timer(5.0, timeout_callback)
            timer.start()
            
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    temperature=0.3,  # 降低温度以提高分类准确性
                    max_tokens=10,
                    timeout=5.0  # 设置5秒超时
                )
                
                timer.cancel()
                
                if timeout_occurred.is_set():
                    print(f"[LLM] 意图分类超时，默认返回chat")
                    return "chat"
                
                intent = response.choices[0].message.content.strip().lower()
                # 清理可能的格式问题
                intent = intent.replace("。", "").replace(".", "").replace("\n", "").strip()
                
                # 验证返回的意图是否有效
                valid_intents = ["schedule", "weather", "news", "festival", "message_board", "chat"]
                if intent not in valid_intents:
                    print(f"[LLM] 分类结果无效: {intent}，默认返回chat")
                    return "chat"
                
                return intent
            except Exception as api_error:
                timer.cancel()
                raise api_error
                
        except TimeoutError:
            print(f"[LLM] 意图分类超时，默认返回chat")
            return "chat"
        except Exception as e:
            print(f"[LLM] 意图分类失败: {e}，默认返回chat")
            import traceback
            traceback.print_exc()
            return "chat"

    def run(self):
        print(f"[LLM] 进程启动 (API模式: {LLM_MODEL_NAME})...")

        client = None
        if not self.mock:
            try:
                from openai import OpenAI
                # 初始化 OpenAI 客户端 (兼容 DeepSeek/MetaHK)
                client = OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL
                )
                print("[LLM] API 客户端初始化成功")
            except Exception as e:
                print(f"[LLM] API 客户端初始化失败: {e}，切换回 Mock 模式")
                self.mock = True

        while True:
            try:
                # 等待 ASR 输入
                data = self.input_queue.get(timeout=1)
                user_text = data["text"]
                emotion = data.get("emotion", "neutral")
                speaker = data.get("speaker", "unknown")

                print(f"[LLM] 收到输入: {user_text} (情绪: {emotion}, 说话人: {speaker})")

                # --- 1. 先进行意图分类 ---
                intent = "chat"  # 默认值
                if not self.mock and client is not None:
                    intent = self.classify_intent(user_text, client)
                    print(f"[LLM] 意图分类结果: {intent}")
                else:
                    print(f"[LLM] Mock模式，跳过意图分类，默认chat")

                # --- 2. 如果是功能意图，返回分类结果给main.py处理 ---
                if intent != "chat":
                    # 返回特殊格式，告知main.py这是功能意图
                    result = {
                        "intent": intent,
                        "text": user_text,
                        "emotion": emotion,
                        "speaker": speaker
                    }
                    self.output_queue.put({"intent_result": json.dumps(result), "end": True})
                    continue

                # --- 3. 如果是正常聊天，生成回复 ---
                speaker_info = f"说话人：{speaker}。" if speaker != "unknown" else ""
                system_prompt = (
                    "你是一个基于树莓派的智能助手'小语'。"
                    f"用户当前情绪：{emotion}。"
                    f"{speaker_info}"
                    "请用简短、亲切的中文回复（50字以内）。"
                    "不要使用Markdown格式，直接输出纯文本。"
                    "回复必须以标点符号结尾（。！？等），不要使用波浪号~。"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]

                # --- 4. 生成/调用 API ---
                if self.mock or client is None:
                    # Mock 模式：模拟打字机效果
                    dummy_response = "网络连接不可用，我现在只能进行模拟对话。"
                    for char in dummy_response:
                        self.output_queue.put({"text_chunk": char, "end": False})
                        time.sleep(0.1)
                    self.output_queue.put({"text_chunk": "", "end": True})
                else:
                    try:
                        # 真实 API 调用 (流式)
                        stream = client.chat.completions.create(
                            model=LLM_MODEL_NAME,
                            messages=messages,
                            stream=True,
                            temperature=0.7,
                            max_tokens=150
                        )

                        full_content = ""
                        sentence_buffer = ""  # 句子缓冲区
                        sentence_endings = ("。", "！", "？", ".", "!", "?", "…")
                        
                        for chunk in stream:
                            # 提取内容 delta
                            if chunk.choices and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                # 过滤掉波浪号~
                                content = content.replace("~", "").replace("～", "")
                                full_content += content
                                sentence_buffer += content
                                
                                # 检查是否有完整句子可以发送
                                while any(end in sentence_buffer for end in sentence_endings):
                                    # 找到第一个句子结束符的位置
                                    min_pos = len(sentence_buffer)
                                    for end in sentence_endings:
                                        pos = sentence_buffer.find(end)
                                        if pos != -1 and pos < min_pos:
                                            min_pos = pos
                                    
                                    # 提取完整句子并发送给 TTS
                                    if min_pos < len(sentence_buffer):
                                        sentence = sentence_buffer[:min_pos + 1]
                                        sentence_buffer = sentence_buffer[min_pos + 1:]
                                        if sentence.strip():
                                            # 立即发送完整句子，实现真正的流式 TTS
                                            self.output_queue.put({"text_chunk": sentence, "end": False})
                                    else:
                                        break

                        # 发送剩余内容（确保有标点符号）
                        if sentence_buffer.strip():
                            # 如果剩余内容没有标点符号，添加句号
                            if not any(end in sentence_buffer for end in sentence_endings):
                                sentence_buffer += "。"
                            # 再次过滤波浪号
                            sentence_buffer = sentence_buffer.replace("~", "").replace("～", "")
                            self.output_queue.put({"text_chunk": sentence_buffer, "end": False})
                        elif full_content and not any(end in full_content for end in sentence_endings):
                            # 如果整个回复都没有标点符号，添加句号
                            self.output_queue.put({"text_chunk": "。", "end": False})
                        
                        # 过滤完整内容中的波浪号
                        full_content = full_content.replace("~", "").replace("～", "")
                        print(f"[LLM] 完整回复: {full_content}")
                        # 发送结束信号
                        self.output_queue.put({"text_chunk": "", "end": True})

                    except Exception as e:
                        print(f"[LLM] API 请求失败: {e}")
                        err_msg = "抱歉，我的大脑连接有点问题。"
                        self.output_queue.put({"text_chunk": err_msg, "end": False})
                        self.output_queue.put({"text_chunk": "", "end": True})

            except queue.Empty:
                continue