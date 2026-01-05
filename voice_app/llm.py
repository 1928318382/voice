import multiprocessing
import queue
import time
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class LLMEngine(multiprocessing.Process):
    """
    认知引擎进程 (API版)
    输入: 用户文本 (Queue)
    输出: 助手回复的流式文本 (Queue)
    """

    def __init__(self, input_queue, output_queue, mock=False):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.mock = mock

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

                # --- 1. 构建 Prompt (使用标准 Messages 格式) ---
                speaker_info = f"说话人：{speaker}。" if speaker != "unknown" else ""
                system_prompt = (
                    "你是一个基于树莓派的智能助手'小派'。"
                    f"用户当前情绪：{emotion}。"
                    f"{speaker_info}"
                    "请用简短、亲切的中文回复（50字以内）。"
                    "不要使用Markdown格式，直接输出纯文本。"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]

                # --- 2. 生成/调用 API ---
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
                        sentence_endings = ("。", "！", "？", ".", "!", "?", "…", "~")
                        
                        for chunk in stream:
                            # 提取内容 delta
                            if chunk.choices and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
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

                        # 发送剩余内容
                        if sentence_buffer.strip():
                            self.output_queue.put({"text_chunk": sentence_buffer, "end": False})
                            
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
