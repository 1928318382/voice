"""
留言板核心模块
负责留言的存储、查询和管理
"""

import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

from app.core.config import BASE_DIR, DATA_DIR


@dataclass
class Message:
    """单条留言记录"""

    id: str
    sender: str  # 发送者（声纹识别ID）
    receiver: str  # 接收者（声纹识别ID）
    content: str  # 留言内容
    timestamp: str  # 留言时间（ISO格式）
    read: bool = False  # 是否已读

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Message":
        return Message(
            id=data["id"],
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            timestamp=data["timestamp"],
            read=data.get("read", False),
        )


class MessageBoardManager:
    """
    留言板管理器：负责内存管理 + JSON 持久化
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.path.join(DATA_DIR, "messages.json")
        self.storage_path = storage_path
        self._messages: List[Message] = []
        self._load()

    def _load(self) -> None:
        """加载留言数据"""
        if not os.path.exists(self.storage_path):
            self._messages = []
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                self._messages = [Message.from_dict(x) for x in raw]
            elif isinstance(raw, dict):
                messages_raw = raw.get("messages", [])
                self._messages = [Message.from_dict(x) for x in messages_raw]
            else:
                self._messages = []
        except Exception as e:
            print(f"[MessageBoard] 加载留言数据失败: {e}")
            self._messages = []

    def _save(self) -> None:
        """保存留言数据"""
        try:
            data = {
                "messages": [msg.to_dict() for msg in self._messages],
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MessageBoard] 保存留言数据失败: {e}")

    def add_message(
        self, sender: str, receiver: str, content: str
    ) -> Message:
        """添加留言"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.now().isoformat(),
            read=False,
        )
        self._messages.append(message)
        self._save()
        print(f"[MessageBoard] 新留言: {sender} -> {receiver}")
        return message

    def get_messages_for_user(
        self, user_id: str, unread_only: bool = False
    ) -> List[Message]:
        """获取指定用户的留言（作为接收者）"""
        messages = [
            msg for msg in self._messages if msg.receiver == user_id
        ]
        if unread_only:
            messages = [msg for msg in messages if not msg.read]
        # 按时间倒序排列（最新的在前）
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        return messages

    def get_all_messages(self, include_read: bool = True) -> List[Message]:
        """获取所有留言（root用户使用）"""
        messages = self._messages
        if not include_read:
            messages = [msg for msg in messages if not msg.read]
        # 按时间倒序排列
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        return messages

    def mark_as_read(self, message_id: str) -> bool:
        """标记留言为已读"""
        for msg in self._messages:
            if msg.id == message_id:
                msg.read = True
                self._save()
                return True
        return False

    def mark_all_as_read(self, user_id: str) -> int:
        """标记指定用户的所有留言为已读"""
        count = 0
        for msg in self._messages:
            if msg.receiver == user_id and not msg.read:
                msg.read = True
                count += 1
        if count > 0:
            self._save()
        return count

    def delete_message(self, message_id: str) -> bool:
        """删除留言"""
        before = len(self._messages)
        self._messages = [msg for msg in self._messages if msg.id != message_id]
        changed = len(self._messages) != before
        if changed:
            self._save()
        return changed

    def get_unread_count(self, user_id: str) -> int:
        """获取指定用户的未读留言数量"""
        return len(
            [
                msg
                for msg in self._messages
                if msg.receiver == user_id and not msg.read
            ]
        )

    def format_messages_for_tts(
        self, messages: List[Message], max_count: int = 5
    ) -> str:
        """将留言列表格式化为TTS文本"""
        if not messages:
            return "您没有留言。"

        # 限制数量
        messages = messages[:max_count]

        lines = []
        for i, msg in enumerate(messages, 1):
            # 解析时间
            try:
                dt = datetime.fromisoformat(msg.timestamp)
                time_str = dt.strftime("%m月%d日 %H:%M")
            except:
                time_str = "未知时间"

            sender_name = msg.sender if msg.sender != "unknown" else "未知用户"
            lines.append(
                f"第{i}条留言，来自{sender_name}，{time_str}，内容是：{msg.content}"
            )

        if len(messages) == 1:
            return lines[0]
        else:
            return "您有{}条留言。".format(len(messages)) + " ".join(lines)

