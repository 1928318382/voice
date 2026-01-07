import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
import re
from enum import Enum
from typing import List, Optional, Dict, Any

from app.core.config import BASE_DIR, DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class ScheduleCategory(str, Enum):
    """日程类别：用药、作息、待办"""

    MEDICATION = "medication"
    ROUTINE = "routine"
    TODO = "todo"


@dataclass
class ScheduleItem:
    """单条日程记录"""

    id: str
    category: ScheduleCategory
    title: str
    time: Optional[str] = None  # 时间描述（不强制格式，但建议使用 HH:MM 或 YYYY-MM-DD HH:MM）
    remark: Optional[str] = None
    # 最近一次触发日期（YYYY-MM-DD），用于避免重复提醒
    last_triggered_date: Optional[str] = None
    # 由 LLM 预生成的更口语化提醒语句（若为空则在运行时用模板生成）
    reminder_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ScheduleItem":
        return ScheduleItem(
            id=data["id"],
            category=ScheduleCategory(data["category"]),
            title=data.get("title", ""),
            time=data.get("time"),
            remark=data.get("remark"),
            last_triggered_date=data.get("last_triggered_date"),
            reminder_text=data.get("reminder_text"),
        )


class ScheduleManager:
    """
    日程管理器：负责内存管理 + JSON 持久化

    支持三个维度：
    - 用药提醒 (medication)
    - 作息提醒 (routine)
    - 待办管理 (todo)
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.path.join(DATA_DIR, "schedules.json")
        self.storage_path = storage_path
        self._items: List[ScheduleItem] = []
        # 简单递增编号（字符串）：1、2、3...
        self._next_id: int = 1
        self._load()

    # ========== 基础存储 ==========

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            self._items = []
            self._next_id = 1
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # 兼容旧格式（纯列表）和新格式（带 next_id 的字典）
            if isinstance(raw, list):
                self._items = [ScheduleItem.from_dict(x) for x in raw]
                nums: List[int] = []
                for it in self._items:
                    try:
                        nums.append(int(it.id))
                    except Exception:
                        continue
                self._next_id = (max(nums) + 1) if nums else (len(self._items) + 1)
            elif isinstance(raw, dict):
                items_raw = raw.get("items", [])
                self._items = [ScheduleItem.from_dict(x) for x in items_raw]
                try:
                    self._next_id = int(raw.get("next_id", len(self._items) + 1))
                except Exception:
                    self._next_id = len(self._items) + 1
            else:
                self._items = []
                self._next_id = 1
        except Exception:
            # 若解析失败，避免整个系统崩溃，直接重置
            self._items = []
            self._next_id = 1

    def _save(self) -> None:
        try:
            data = {
                "next_id": self._next_id,
                "items": [item.to_dict() for item in self._items],
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            # 保存失败时不抛出，主流程不受影响
            pass

    # ========== CRUD 接口 ==========

    def add_item(
        self,
        category: ScheduleCategory,
        title: str,
        time_str: Optional[str] = None,
        remark: Optional[str] = None,
        reminder_text: Optional[str] = None,
    ) -> ScheduleItem:
        item = ScheduleItem(
            id=str(self._next_id),
            category=category,
            title=title,
            time=time_str,
            remark=remark,
            reminder_text=reminder_text,
        )
        self._items.append(item)
        self._next_id += 1
        self._save()
        return item

    def delete_item(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [it for it in self._items if it.id != item_id]
        changed = len(self._items) != before
        if changed:
            self._save()
        return changed

    def update_item(
        self,
        item_id: str,
        title: Optional[str] = None,
        time_str: Optional[str] = None,
        remark: Optional[str] = None,
    ) -> bool:
        for it in self._items:
            if it.id == item_id:
                if title is not None:
                    it.title = title
                if time_str is not None:
                    it.time = time_str
                if remark is not None:
                    it.remark = remark
                self._save()
                return True
        return False

    def list_items(
        self, category: Optional[ScheduleCategory] = None
    ) -> List[ScheduleItem]:
        if category is None:
            return list(self._items)
        return [it for it in self._items if it.category == category]

    # ========== 定时提醒相关 ==========

    def _parse_time(self, time_str: str, today: date) -> Optional[datetime]:
        """
        解析时间字符串：
        - 若为 'YYYY-MM-DD HH:MM'，视为一次性提醒
        - 若为 'HH:MM'，视为每天该时间触发一次
        其它格式暂不做主动提醒，只存储文本。
        """
        time_str = time_str.strip()
        # 一次性：完整日期时间
        try:
            if len(time_str) == 16 and time_str[4] == "-" and time_str[7] == " " and time_str[10] == ":":
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except Exception:
            pass

        # 每日：HH:MM
        try:
            if len(time_str) == 5 and time_str[2] == ":":
                hh, mm = time_str.split(":")
                t = dtime(hour=int(hh), minute=int(mm))
                return datetime.combine(today, t)
        except Exception:
            pass

        return None

    def get_due_items(self, now: datetime, tolerance_minutes: int = 1) -> List[ScheduleItem]:
        """
        返回“到点需要提醒”的日程。

        要求：
        - time 字段可被解析为标准时间（见 _parse_time）
        - 当前时间与目标时间相差不超过 tolerance_minutes
        - 且今天还未触发（基于 last_triggered_date 防抖）
        """
        due: List[ScheduleItem] = []
        today_str = now.date().isoformat()

        for it in self._items:
            if not it.time:
                continue

            target_dt = self._parse_time(it.time, now.date())
            if target_dt is None:
                continue

            # 只在当天考虑每日提醒
            if target_dt.date() != now.date():
                # 若是一次性提醒，过了日期就不再触发
                continue

            delta = abs((now - target_dt).total_seconds()) / 60.0
            if delta <= tolerance_minutes:
                # 今日尚未触发
                if it.last_triggered_date != today_str:
                    it.last_triggered_date = today_str
                    due.append(it)

        if due:
            self._save()
        return due


# ========== 使用 LLM 做自然语句解析（方案 A） ==========


def format_time_for_speech(time_str: str) -> str:
    """
    将ISO格式时间转换为口语化的中文表达
    例如: "2026-01-07 13:30" -> "一月七号十三点三十"
    """
    if not time_str:
        return ""

    # 处理完整日期时间格式 YYYY-MM-DD HH:MM
    match_full = re.match(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})", time_str)
    if match_full:
        year, month, day, hour, minute = match_full.groups()
        # 只显示月日，如果是今年就不显示年
        current_year = str(datetime.now().year)
        date_part = ""
        if year != current_year:
            date_part = f"{int(year)}年"  # 简化为后两位，如2026->26年
        date_part += f"{int(month)}月{int(day)}号"

        # 时间部分
        hour_int = int(hour)
        if hour_int == 0:
            time_part = "零点"
        elif hour_int < 12:
            time_part = f"上午{hour_int}点"
        elif hour_int == 12:
            time_part = "中午12点"
        else:
            time_part = f"下午{hour_int-12}点"

        if minute != "00":
            time_part += f"{int(minute)}"

        return f"{date_part}{time_part}"

    # 处理只有时间格式 HH:MM
    match_time = re.match(r"(\d{2}):(\d{2})", time_str)
    if match_time:
        hour, minute = match_time.groups()
        hour_int = int(hour)
        if hour_int == 0:
            time_part = "零点"
        elif hour_int < 12:
            time_part = f"上午{hour_int}点"
        elif hour_int == 12:
            time_part = "中午12点"
        else:
            time_part = f"下午{hour_int-12}点"

        if minute != "00":
            time_part += f"{int(minute)}"

        return time_part

    # 如果格式不匹配，返回原字符串
    return time_str


class ScheduleLLMParser:
    """
    使用与对话相同的 LLM 后端，将自然语言句子解析为结构化日程指令。

    只做“理解”，不负责真正写入数据库。
    """

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        except Exception:
            self._client = None
        return self._client

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        调用 LLM，将一句话解析成 JSON：
        {
          "intent": "add" | "delete" | "update" | "query" | "none",
          "category": "medication" | "routine" | "todo" | null,
          "time": "HH:MM" 或 "YYYY-MM-DD HH:MM" 或 "",
          "title": "吃降压药" 等,
          "raw": 原始句子
        }
        """
        client = self._ensure_client()
        if client is None:
            return None

        # 给 LLM 明确“当前时间”，方便它把“明天中午”之类解析为具体时间
        now = datetime.now()
        now_date = now.strftime("%Y-%m-%d")
        now_time = now.strftime("%H:%M")

        system_prompt = (
            "你是一个只负责“解析指令”的助手，不负责闲聊。\n"
            f"当前日期是 {now_date}，当前时间是 {now_time}（24小时制）。\n"
            "用户会用中文说与日程有关的话，你需要把它解析成一个 JSON，字段包括：\n"
            'intent: "add" | "delete" | "update" | "query" | "none"\n'
            'category: "medication" | "routine" | "todo" | null\n'
            'time: 若能确定具体时间，用24小时制 \"HH:MM\" 或 \"YYYY-MM-DD HH:MM\"，'
            "例如“明天中午”可解析为“YYYY-MM-DD 12:00”；如果无法确定具体时间，用空字符串 \"\"。\n"
            "title: 这条日程的内容（简短中文）。\n"
            'item_id: 如果是删除或修改操作，请从用户话语中提取日程编号（纯数字），否则用 null。\n'
            'update_field: 如果是修改操作，指定要修改的字段："time" | "title"，否则用 null。\n'
            'update_value: 如果是修改操作，要修改成的新值，否则用 null。\n'
            "只返回 JSON，本身不能包含任何中文解释。"
        )

        user_prompt = (
            "请解析下面这句话，推断与日程相关的意图：\n"
            f"{text}\n\n"
            "如果是新增或修改提醒，请尽量从句子中抽取合理的时间；\n"
            "如果是修改操作，请判断是要修改时间还是标题：\n"
            "- 如果提到时间相关词（几点、上午、下午等），设置 update_field 为 \"time\"\n"
            "- 如果提到内容、标题相关词（改为、改成、内容是、标题为等），设置 update_field 为 \"title\"\n"
            "如果与日程无关，intent 请填 \"none\"。"
        )

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            content = resp.choices[0].message.content.strip()
            # 保险起见，只取最外层大括号之间的内容
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            json_str = content[start : end + 1]
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None
            data.setdefault("raw", text)
            return data
        except Exception:
            return None

    def generate_reminder_text(
        self,
        category: Optional[ScheduleCategory],
        title: str,
        time_str: Optional[str],
    ) -> Optional[str]:
        """
        让 LLM 生成一条简短、口语化的提醒语句。
        """
        client = self._ensure_client()
        if client is None:
            return None

        cat_cn = ""
        if category == ScheduleCategory.MEDICATION:
            cat_cn = "用药"
        elif category == ScheduleCategory.ROUTINE:
            cat_cn = "作息"
        elif category == ScheduleCategory.TODO:
            cat_cn = "待办"

        system_prompt = (
            "你是一个帮忙润色提醒话术的小助手。\n"
            "请根据给定的类别、时间和内容，生成一句简短、口语化的中文提醒，用于语音播报。\n"
            "要求：\n"
            "1. 20 个字左右，最多不超过 30 字。\n"
            "2. 语气自然、亲切，例如“现在到吃药时间啦，记得睡前吃降压药。”\n"
            "3. 不要使用 Markdown，只输出一句话。\n"
        )

        user_prompt = f"类别：{cat_cn or '一般事务'}\n时间：{time_str or '未指定'}\n内容：{title}\n请生成一句提醒话。"

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 只取首行，避免 LLM 啰嗦
            return text.splitlines()[0] if text else None
        except Exception:
            return None


# ========== 中文命令 + 自然语句解析 ==========


class ScheduleCommandHandler:
    """
    将中文文本（命令式 + 自然语句）映射到 ScheduleManager 的 CRUD 操作。

    1. 优先解析显式指令，以“日程 ...”开头的文本；
    2. 再尝试解析更自然的说法，例如：
       - 记一下，明天早上八点吃降压药
       - 帮我提醒，晚上十点睡觉
       - 安排一下，下午三点写报告
    """

    CATEGORY_MAP = {
        "用药": ScheduleCategory.MEDICATION,
        "吃药": ScheduleCategory.MEDICATION,
        "药": ScheduleCategory.MEDICATION,
        "作息": ScheduleCategory.ROUTINE,
        "起床": ScheduleCategory.ROUTINE,
        "睡觉": ScheduleCategory.ROUTINE,
        "待办": ScheduleCategory.TODO,
        "任务": ScheduleCategory.TODO,
        "todo": ScheduleCategory.TODO,
    }

    # 触发“新增日程”的自然表达关键词
    NATURAL_ADD_PREFIX = (
        "记一下",
        "帮我记",
        "提醒我",
        "帮我提醒",
        "安排一下",
        "帮我安排",
        "添加一个",
        "加一个",
    )

    def __init__(self, manager: Optional[ScheduleManager] = None, llm_parser: Optional[ScheduleLLMParser] = None):
        self.manager = manager or ScheduleManager()
        self.llm_parser = llm_parser or ScheduleLLMParser()

    def handle(self, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            return None

        # 支持“删除日程1”这种说法：直接解析为删除命令
        if text.startswith("删除日程"):
            item_id = text.replace("删除日程", "", 1).strip()
            if item_id:
                return self._handle_delete([item_id])
            # 没给编号就当作普通文本

        # 1) 显式命令：以“日程”开头 —— 本地规则，优先处理
        if text.startswith("日程"):
            return self._handle_explicit(text)

        # 2) 交给 LLM 做解析：希望由 LLM 来理解“明天中午”等自然表达
        parsed = self._handle_llm(text)
        if parsed:
            return parsed

        # 3) 兜底：再尝试一遍简单自然规则
        natural = self._handle_natural(text)
        if natural:
            return natural

        return None

    # ------------------------------------------------------------------
    # 显式命令解析：沿用原有“日程 新增/删除/修改/查询 ...”语法
    # ------------------------------------------------------------------

    def _handle_explicit(self, text: str) -> str:
        cmd = text[2:].strip()
        if not cmd:
            return "你可以说：日程 新增 / 删除 / 修改 / 查询。"

        parts = cmd.split()
        if not parts:
            return "没有听清具体要做什么，请再说一遍。"

        action = parts[0]

        # 兼容“日程删除1”这种没有空格的写法
        if len(parts) == 1 and action.startswith("删除"):
            item_id = action.replace("删除", "", 1).strip()
            if item_id:
                return self._handle_delete([item_id])
            # 没有编号则继续走下面的分支

        if action in ("新增", "添加"):
            return self._handle_add(parts[1:])
        if action in ("删除", "移除"):
            return self._handle_delete(parts[1:])
        if action in ("修改", "更新"):
            return self._handle_update(parts[1:])
        if action in ("查询", "查看", "列出"):
            return self._handle_query(parts[1:])

        return "目前支持：日程 新增 / 删除 / 修改 / 查询。你可以再试一次。"

    # ------------------------------------------------------------------
    # 自然语句解析（简单启发式）
    # ------------------------------------------------------------------

    def _handle_natural(self, text: str) -> Optional[str]:
        """尝试从更自然的中文句子里，推断出“新增”操作。"""
        lowered = text.replace("，", ",").replace("。", ",")

        # 必须包含“记 / 提醒 / 安排 / 加一个 / 添加一个”之类的触发词，否则不抢占普通对话
        if not any(prefix in lowered for prefix in self.NATURAL_ADD_PREFIX):
            return None

        # 简单切掉前导触发词，比如“记一下，”“帮我提醒”
        for prefix in self.NATURAL_ADD_PREFIX:
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :].lstrip("，,。 ")
                break

        if not lowered:
            return "你想让我帮你记哪一类事情？比如吃药、作息或者待办？"

        # 粗略判断类别：看是否出现“吃药、药、起床、睡觉、写报告”等词
        category = None
        if any(k in lowered for k in ("吃药", "用药", "药")):
            category = ScheduleCategory.MEDICATION
        elif any(k in lowered for k in ("起床", "睡觉", "午休", "早起", "早睡", "晚睡")):
            category = ScheduleCategory.ROUTINE
        elif any(k in lowered for k in ("写报告", "作业", "开会", "任务", "待办", "todo")):
            category = ScheduleCategory.TODO

        # 简化：时间描述 = 句子里前半段的“时间相关词”，标题 = 整个剩余句子
        # 在这里我们不过度切分，直接把完整句子当成标题，同时把“可能的时间短语”先留在 time_str 里
        time_str = None
        title = lowered

        # 一个非常粗糙的时间短语抽取：看有没有“早上/晚上/下午/明天/后天”等
        time_keywords = ("早上", "上午", "中午", "下午", "晚上", "明天", "后天", "每天", "每晚", "每周")
        for kw in time_keywords:
            if kw in lowered:
                # 以这个关键词开头到下一个逗号，粗略截为 time_str
                idx = lowered.index(kw)
                seg = lowered[idx:]
                if "," in seg:
                    seg = seg.split(",")[0]
                time_str = seg.strip()
                break

        if category is None:
            # 你的需求：所有非用药、非作息的情况，默认归为“待办”
            category = ScheduleCategory.TODO

        # 使用 LLM 预先生成一条提醒语句（失败则为 None，后续有模板兜底）
        reminder = self.llm_parser.generate_reminder_text(category, title or "未命名", time_str)

        item = self.manager.add_item(
            category=category,
            title=title or "未命名",
            time_str=time_str,
            reminder_text=reminder,
        )

        if category == ScheduleCategory.MEDICATION:
            cat_cn = "用药"
        elif category == ScheduleCategory.ROUTINE:
            cat_cn = "作息"
        else:
            cat_cn = "待办"

        return f"好的，已经帮你记下一个{cat_cn}：{title}，编号 {item.id}。"

    # ------------------------------------------------------------------
    # 3) 调用 LLM 做更复杂自然语句的解析
    # ------------------------------------------------------------------

    def _handle_llm(self, text: str) -> Optional[str]:
        """调用 LLM 解析更自由的自然语句。"""
        data = self.llm_parser.parse(text)
        if not data:
            return None

        intent = data.get("intent") or "none"
        if intent == "none":
            return None

        # category 映射
        cat_str = data.get("category")
        category = None
        if cat_str in ("medication", "routine", "todo"):
            category = ScheduleCategory(cat_str)

        # 现在支持通过 LLM 新增、删除、修改和查询
        if intent == "add":
            title = (data.get("title") or "").strip() or text
            time_str = (data.get("time") or "").strip() or None
            if category is None:
                # 你的需求：所有非用药、非作息的情况，默认归为“待办”
                category = ScheduleCategory.TODO

            reminder = self.llm_parser.generate_reminder_text(category, title, time_str)

            item = self.manager.add_item(
                category=category,
                title=title,
                time_str=time_str,
                reminder_text=reminder,
            )

            if category == ScheduleCategory.MEDICATION:
                cat_cn = "用药"
            elif category == ScheduleCategory.ROUTINE:
                cat_cn = "作息"
            else:
                cat_cn = "待办"

            when = f"，时间 {time_str}" if time_str else ""
            return f"好的，已经根据你的话帮你新增一个{cat_cn}日程：{title}{when}，编号 {item.id}。"

        if intent == "query":
            # 如果 LLM 给出了类别，就按类别查询；否则查全部
            items = self.manager.list_items(category=category)
            if not items:
                return "目前还没有相关日程。"

            prefix = ""
            if category == ScheduleCategory.MEDICATION:
                prefix = "用药"
            elif category == ScheduleCategory.ROUTINE:
                prefix = "作息"
            elif category == ScheduleCategory.TODO:
                prefix = "待办"

            # 语音友好格式：先说总数，再逐条说明
            total_count = len(items)
            display_count = min(5, total_count)

            if prefix:
                head = f"你现在有{total_count}条{prefix}日程。"
            else:
                head = f"你现在有{total_count}条日程。"

            lines = []
            for i, it in enumerate(items[:display_count], 1):
                time_part = f"{format_time_for_speech(it.time)}，" if it.time else ""
                lines.append(f"第{i}条，{time_part}{it.title}，编号是{it.id}")

            result = head + " ".join(lines)

            if total_count > 5:
                result += f" 另外还有{total_count - 5}条，暂时就不一一念了。"
            elif total_count > 0:
                result += "。"  # 结尾句号

            return result

        if intent == "delete":
            item_id = data.get("item_id")
            if item_id is None:
                return "我没有听出你要删除哪个日程，你可以说：删除编号1的日程。"
            try:
                item_id = int(item_id)
                ok = self.manager.delete_item(str(item_id))
                if ok:
                    return f"好的，已经删除编号为 {item_id} 的日程。"
                else:
                    return f"没有找到编号为 {item_id} 的日程。"
            except ValueError:
                return "日程编号应该是数字，你可以说：删除编号1的日程。"

        if intent == "update":
            item_id = data.get("item_id")
            update_field = data.get("update_field")
            update_value = data.get("update_value")

            if item_id is None:
                return "我没有听出你要修改哪个日程，你可以说：把编号1的日程时间改成早上8点。"
            if update_field is None or update_value is None:
                return "我没有听出你要修改什么内容，你可以说：把编号1的日程时间改成早上8点，或者把编号1的日程标题改为提醒我休息。"

            try:
                item_id = int(item_id)
                kwargs = {}
                if update_field == "time":
                    kwargs["time_str"] = update_value
                elif update_field == "title":
                    kwargs["title"] = update_value
                else:
                    return "目前只能修改时间或标题。"

                ok = self.manager.update_item(str(item_id), **kwargs)
                if ok:
                    return f"好的，已经更新编号 {item_id} 的日程。"
                else:
                    return f"没有找到编号为 {item_id} 的日程。"
            except ValueError:
                return "日程编号应该是数字。"

        return None

    # ------------------------------------------------------------------
    # 下面是增删改查的具体实现（与原始版本基本一致）
    # ------------------------------------------------------------------

    def _parse_category(self, token: str) -> Optional[ScheduleCategory]:
        return self.CATEGORY_MAP.get(token)

    def _handle_add(self, parts: List[str]) -> str:
        if not parts:
            return "请说明是新增用药、作息还是待办，比如：日程 新增 用药 明天早上8点 吃降压药。"

        cat_token = parts[0]
        category = self._parse_category(cat_token)
        if not category:
            return "没听懂是用药、作息还是待办，请再说一次。"

        # 简化：第二个分段尽量当时间，其余合并为标题
        time_str = None
        title = ""
        if len(parts) == 1:
            return "请再补充时间和内容，比如：日程 新增 用药 每天早上8点 吃降压药。"
        elif len(parts) >= 2:
            time_str = parts[1]
            if len(parts) >= 3:
                title = " ".join(parts[2:])
            else:
                title = parts[1]
                time_str = None

        reminder = self.llm_parser.generate_reminder_text(category, title or "未命名", time_str)

        item = self.manager.add_item(
            category=category,
            title=title or "未命名",
            time_str=time_str,
            reminder_text=reminder,
        )
        return f"好的，已为你新增一条{cat_token}日程，编号 {item.id}。"

    def _handle_delete(self, parts: List[str]) -> str:
        if not parts:
            return "请提供要删除的日程编号，比如：日程 删除 1234abcd。"
        item_id = parts[0]
        ok = self.manager.delete_item(item_id)
        if ok:
            return f"好的，已经删除编号为 {item_id} 的日程。"
        return f"没有找到编号为 {item_id} 的日程。"

    def _handle_update(self, parts: List[str]) -> str:
        if len(parts) < 3:
            return "修改需要提供编号、字段和内容，比如：日程 修改 1234abcd 时间 明天早上8点。"

        item_id = parts[0]
        field = parts[1]
        value = " ".join(parts[2:])

        kwargs: Dict[str, Any] = {}
        if field in ("时间", "time"):
            kwargs["time_str"] = value
        elif field in ("标题", "名称", "title"):
            kwargs["title"] = value
        elif field in ("备注", "说明", "remark"):
            kwargs["remark"] = value
        else:
            return "目前只能修改时间、标题或备注。"

        ok = self.manager.update_item(item_id, **kwargs)
        if ok:
            return f"好的，已经更新编号 {item_id} 的日程。"
        return f"没有找到编号为 {item_id} 的日程。"

    def _handle_query(self, parts: List[str]) -> str:
        category: Optional[ScheduleCategory] = None
        if parts:
            category = self._parse_category(parts[0])
        items = self.manager.list_items(category=category)
        if not items:
            return "目前还没有相关日程。"

        prefix = ""
        if category == ScheduleCategory.MEDICATION:
            prefix = "用药"
        elif category == ScheduleCategory.ROUTINE:
            prefix = "作息"
        elif category == ScheduleCategory.TODO:
            prefix = "待办"

        # 语音友好格式：先说总数，再逐条说明
        total_count = len(items)
        display_count = min(5, total_count)

        if prefix:
            head = f"你现在有{total_count}条{prefix}日程。"
        else:
            head = f"你现在有{total_count}条日程。"

        lines = []
        for i, it in enumerate(items[:display_count], 1):
            time_part = f"{format_time_for_speech(it.time)}，" if it.time else ""
            lines.append(f"第{i}条，{time_part}{it.title}，编号是{it.id}")

        result = head + " ".join(lines)

        if total_count > 5:
            result += f" 你想让我把剩下的{total_count - 5}条也念完吗？"
            # 返回特殊格式，表示有后续内容等待用户确认
            return f"PARTIAL_QUERY:{result}:{total_count}:{display_count}"
        elif total_count > 0:
            result += "。"  # 结尾句号

        return result


if __name__ == "__main__":
    # 简单命令行测试
    mgr = ScheduleManager()
    handler = ScheduleCommandHandler(mgr)
    print("输入示例：")
    print("  日程 新增 用药 08:00 每天早上八点吃降压药")
    print("  记一下，明天早上八点吃降压药")
    print("  删除编号1的日程")
    print("  把编号2的日程时间改成下午3点")
    print("  把编号3的日程标题改为提醒我休息")
    print("  查看我的日程")
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if not s:
            continue
        if s in ("q", "quit", "exit"):
            break
        resp = handler.handle(s)
        print("=>", resp or "(未匹配为日程命令)")


