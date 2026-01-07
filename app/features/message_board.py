"""
留言板功能对外接口
"""

from typing import Optional
from .message_board_core import MessageBoardManager


class MessageBoardCommandHandler:
    """
    留言板命令处理器
    支持留言查看、发送留言等功能
    """

    def __init__(self, speaker_recognizer=None):
        self.manager = MessageBoardManager()
        self.speaker_recognizer = speaker_recognizer

    def notify_user_messages(self, user_id: str) -> Optional[str]:
        """
        当识别到用户时，自动检查并通知用户有未读留言
        返回通知文本，如果没有留言则返回None
        """
        if user_id == "unknown":
            return None

        try:
            unread_count = self.manager.get_unread_count(user_id)
            if unread_count > 0:
                messages = self.manager.get_messages_for_user(user_id, unread_only=True)
                # 只显示最新的3条
                formatted = self.manager.format_messages_for_tts(messages, max_count=3)
                # 标记为已读
                self.manager.mark_all_as_read(user_id)
                return f"您有{unread_count}条未读留言。{formatted}"
        except Exception as e:
            print(f"[MessageBoard] 通知用户留言失败: {e}")
        
        return None

    def handle(self, text: str, speaker: str = "unknown") -> Optional[str]:
        """
        处理留言板相关指令
        支持的指令：
        - "查看留言" / "读留言" / "我的留言" - 查看自己的留言
        - "给XX留言XXX" / "给XX留个言" - 发送留言给指定用户
        """
        text = text.strip()
        if not text:
            return None

        text_lower = text.lower()

        # 检查是否是查看留言的指令
        if any(keyword in text_lower for keyword in ["查看留言", "读留言", "我的留言", "留言", "有留言吗"]):
            if speaker == "unknown":
                return "抱歉，无法识别您的身份，无法查看留言。"
            return self._handle_view_messages(speaker)

        # 检查是否是发送留言的指令
        if any(keyword in text_lower for keyword in ["给", "留言", "留个言"]):
            return self._handle_send_message(text, speaker)

        return None

    def _handle_view_messages(self, user_id: str) -> str:
        """处理查看留言请求"""
        try:
            messages = self.manager.get_messages_for_user(user_id, unread_only=False)
            if not messages:
                return "您目前没有留言。"
            
            # 格式化并返回
            formatted = self.manager.format_messages_for_tts(messages, max_count=5)
            # 标记为已读
            self.manager.mark_all_as_read(user_id)
            return formatted
        except Exception as e:
            print(f"[MessageBoard] 查看留言失败: {e}")
            return "抱歉，查看留言时出现错误。"

    def _handle_send_message(self, text: str, sender: str) -> str:
        """处理发送留言请求"""
        try:
            # 简单的关键词提取
            # 格式：给XX留言XXX 或 给XX留个言XXX
            text_lower = text.lower()
            
            # 尝试提取接收者和内容
            receiver = None
            content = ""
            
            # 检查是否有"给"字
            if "给" in text:
                parts = text.split("给", 1)
                if len(parts) > 1:
                    rest = parts[1]
                    # 尝试找到"留言"或"留个言"
                    if "留言" in rest:
                        msg_parts = rest.split("留言", 1)
                        if len(msg_parts) >= 1:
                            receiver_candidate = msg_parts[0].strip()
                            if len(msg_parts) > 1:
                                content = msg_parts[1].strip()
                            else:
                                content = ""
                            receiver = receiver_candidate
                    elif "留个言" in rest:
                        msg_parts = rest.split("留个言", 1)
                        if len(msg_parts) >= 1:
                            receiver_candidate = msg_parts[0].strip()
                            if len(msg_parts) > 1:
                                content = msg_parts[1].strip()
                            else:
                                content = ""
                            receiver = receiver_candidate

            # 如果没有找到接收者，尝试从声纹识别器获取用户列表
            if not receiver or receiver == "":
                if self.speaker_recognizer:
                    users = self.speaker_recognizer.get_user_list()
                    if users:
                        # 尝试匹配用户名
                        for user in users:
                            if user in text:
                                receiver = user
                                # 提取内容（去掉用户名和"留言"等关键词）
                                content = text.replace(f"给{receiver}", "").replace("留言", "").replace("留个言", "").strip()
                                break

            if not receiver or receiver == "":
                return "请告诉我留言给谁，比如：给张三留言你好"

            if not content or content == "":
                return f"请告诉我留言内容，比如：给{receiver}留言你好"

            # 检查接收者是否存在（如果声纹识别器可用）
            if self.speaker_recognizer:
                users = self.speaker_recognizer.get_user_list()
                if receiver not in users and receiver != "unknown":
                    return f"抱歉，用户{receiver}未注册，无法发送留言。"

            # 发送留言
            sender_id = sender if sender != "unknown" else "unknown"
            message = self.manager.add_message(sender_id, receiver, content)
            return f"已成功给{receiver}留言：{content}"

        except Exception as e:
            print(f"[MessageBoard] 发送留言失败: {e}")
            return "抱歉，发送留言时出现错误。"

