"""
节日提醒核心模块
支持传统节日、节气提醒，以及自定义节日
"""

import json
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import sys
import os
# 如有必要添加 sys.path，但通常由 server 负责。
# 为保险起见，如果作为模块导入，使用全路径：
from app.core.config import BASE_DIR, DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


@dataclass
class FestivalItem:
    """节日条目"""
    id: str
    name: str
    type: str  # "traditional" (传统节日), "solar_term" (节气), "custom" (自定义)
    date_type: str  # "fixed" (固定日期), "lunar" (农历), "custom_date" (自定义日期)
    month: Optional[int] = None
    day: Optional[int] = None
    lunar_month: Optional[int] = None
    lunar_day: Optional[int] = None
    custom_date: Optional[str] = None  # YYYY-MM-DD格式
    reminder_type: str = "text"  # "text" (文字), "song" (唱歌), "poem" (诗朗诵), "custom" (自定义)
    reminder_content: Optional[str] = None
    is_active: bool = True


class FestivalManager:
    """节日管理器"""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.path.join(DATA_DIR, "festivals.json")
        self.storage_path = storage_path
        self._festivals: List[FestivalItem] = []
        self._load()

        # 如果是第一次运行，初始化内置节日
        if not self._festivals:
            self._initialize_builtin_festivals()

    def _load(self) -> None:
        """加载节日数据"""
        if not os.path.exists(self.storage_path):
            self._festivals = []
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._festivals = [FestivalItem(**item) for item in data]
        except Exception:
            self._festivals = []

    def _save(self) -> None:
        """保存节日数据"""
        try:
            data = [asdict(festival) for festival in self._festivals]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _initialize_builtin_festivals(self) -> None:
        """初始化内置节日数据"""
        builtin_festivals = [
            # 传统节日
            FestivalItem("spring_festival", "春节", "traditional", "lunar", lunar_month=1, lunar_day=1,
                        reminder_content="新春快乐！祝您新的一年万事如意，身体健康！"),
            FestivalItem("lantern_festival", "元宵节", "traditional", "lunar", lunar_month=1, lunar_day=15,
                        reminder_content="元宵节快乐！祝您阖家团圆，和美幸福！"),
            FestivalItem("qingming", "清明节", "traditional", "fixed", month=4, day=5,
                        reminder_content="清明节到了，愿您缅怀先人，寄托思念。"),
            FestivalItem("dragon_boat", "端午节", "traditional", "lunar", lunar_month=5, lunar_day=5,
                        reminder_content="端午节安康！愿您幸福美满，阖家安康！"),
            FestivalItem("mid_autumn", "中秋节", "traditional", "lunar", lunar_month=8, lunar_day=15,
                        reminder_content="中秋节快乐！愿您月满常圆，人圆家圆！"),
            FestivalItem("double_ninth", "重阳节", "traditional", "lunar", lunar_month=9, lunar_day=9,
                        reminder_content="重阳节快乐！祝您健康长寿，福禄双全！"),

            # 节气（简化版，只包含主要节气）
            FestivalItem("spring_begins", "立春", "solar_term", "fixed", month=2, day=4,
                        reminder_content="立春到了，万物始生，愿您新的一年充满生机和希望！"),
            FestivalItem("summer_begins", "立夏", "solar_term", "fixed", month=5, day=6,
                        reminder_content="立夏了，夏天来临，愿您清凉度夏，活力满满！"),
            FestivalItem("autumn_begins", "立秋", "solar_term", "fixed", month=8, day=8,
                        reminder_content="立秋了，秋天来临，愿您秋高气爽，心情愉悦！"),
            FestivalItem("winter_begins", "立冬", "solar_term", "fixed", month=11, day=8,
                        reminder_content="立冬了，冬天来临，愿您温暖如春，平安喜乐！"),

            # 现代节日
            FestivalItem("new_year", "元旦", "traditional", "fixed", month=1, day=1,
                        reminder_content="元旦快乐！新的一年开始，祝您万事顺意！"),
            FestivalItem("valentine", "情人节", "traditional", "fixed", month=2, day=14,
                        reminder_content="情人节快乐！愿您爱情甜蜜，幸福美满！"),
            FestivalItem("womens_day", "妇女节", "traditional", "fixed", month=3, day=8,
                        reminder_content="三八妇女节快乐！祝天下女性节日快乐，永远年轻美丽！"),
            FestivalItem("labor_day", "劳动节", "traditional", "fixed", month=5, day=1,
                        reminder_content="五一劳动节快乐！感谢您的辛勤付出，愿您工作顺利！"),
            FestivalItem("childrens_day", "儿童节", "traditional", "fixed", month=6, day=1,
                        reminder_content="六一儿童节快乐！愿孩子们健康成长，快乐每一天！"),
            FestivalItem("national_day", "国庆节", "traditional", "fixed", month=10, day=1,
                        reminder_content="国庆节快乐！祝福伟大祖国繁荣昌盛！"),
        ]

        self._festivals = builtin_festivals
        self._save()

    def add_custom_festival(self, name: str, date_str: str, reminder_type: str = "text",
                          reminder_content: Optional[str] = None) -> FestivalItem:
        """添加自定义节日"""
        try:
            # 解析日期
            if date_str.count('-') == 2:  # YYYY-MM-DD格式
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                festival = FestivalItem(
                    id=f"custom_{len(self._festivals)}",
                    name=name,
                    type="custom",
                    date_type="custom_date",
                    custom_date=date_str,
                    reminder_type=reminder_type,
                    reminder_content=reminder_content or f"{name}快乐！"
                )
            else:
                raise ValueError("日期格式应为YYYY-MM-DD")

            self._festivals.append(festival)
            self._save()
            return festival

        except Exception as e:
            raise ValueError(f"添加自定义节日失败: {e}")

    def update_festival_reminder(self, festival_id: str, reminder_type: str,
                               reminder_content: Optional[str] = None) -> bool:
        """更新节日提醒方式"""
        for festival in self._festivals:
            if festival.id == festival_id:
                festival.reminder_type = reminder_type
                festival.reminder_content = reminder_content
                self._save()
                return True
        return False

    def get_today_festivals(self, today: date) -> List[FestivalItem]:
        """获取当天的节日"""
        today_festivals = []

        for festival in self._festivals:
            if not festival.is_active:
                continue

            is_today = False

            if festival.date_type == "fixed" and festival.month and festival.day:
                # 固定日期节日
                is_today = (festival.month == today.month and festival.day == today.day)

            elif festival.date_type == "custom_date" and festival.custom_date:
                # 自定义日期节日
                try:
                    festival_date = datetime.strptime(festival.custom_date, "%Y-%m-%d").date()
                    is_today = (festival_date.month == today.month and festival_date.day == today.day)
                except:
                    pass

            # 注意：农历节日需要额外的农历转换库，这里暂时简化处理
            elif festival.date_type == "lunar":
                # 简单近似：农历节日按公历大致对应（实际需要农历转换）
                lunar_approximations = {
                    (1, 1): (1, 21),    # 春节 ≈ 1月21日
                    (1, 15): (2, 5),    # 元宵节 ≈ 2月5日
                    (5, 5): (5, 28),    # 端午节 ≈ 5月28日
                    (8, 15): (9, 10),   # 中秋节 ≈ 9月10日
                    (9, 9): (10, 4),    # 重阳节 ≈ 10月4日
                }
                approx_date = lunar_approximations.get((festival.lunar_month, festival.lunar_day))
                if approx_date:
                    is_today = (approx_date[0] == today.month and approx_date[1] == today.day)

            if is_today:
                today_festivals.append(festival)

        return today_festivals

    def get_festival_by_id(self, festival_id: str) -> Optional[FestivalItem]:
        """根据ID获取节日"""
        for festival in self._festivals:
            if festival.id == festival_id:
                return festival
        return None

    def list_all_festivals(self) -> List[FestivalItem]:
        """列出所有节日"""
        return list(self._festivals)


class FestivalLLMParser:
    """使用LLM解析节日相关指令"""

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

    def parse_festival_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        解析节日相关指令
        返回: {"intent": "add_custom"|"update_reminder"|"query_festivals",
               "festival_name": "...", "date": "...", "reminder_type": "...", ...}
        """
        client = self._ensure_client()
        if client is None:
            return None

        system_prompt = (
            "你是一个节日提醒助手，请解析用户关于节日的指令。\n"
            "用户可能想添加自定义节日、修改提醒方式，或查询节日信息。\n"
            "\n"
            "节日类型包括：传统节日、生日、纪念日（如入团纪念日、结婚纪念日等）、周年纪念等。\n"
            "\n"
            "返回JSON格式：\n"
            '{"intent": "add_custom" | "update_reminder" | "query_festivals", \n'
            ' "festival_name": "节日名称（如：入团纪念日、我的生日等）", \n'
            ' "date": "YYYY-MM-DD格式的日期（如：2026-01-08）", \n'
            ' "reminder_type": "text" | "song" | "poem" | "custom", \n'
            ' "reminder_content": "提醒内容或歌词等"}\n'
            "\n"
            "示例：\n"
            "输入：\"一月八号是一个特殊的日子它是我的入团纪念日把一月八号设定为我的入团纪念日\"\n"
            "输出：{\"intent\": \"add_custom\", \"festival_name\": \"入团纪念日\", \"date\": \"2026-01-08\", \"reminder_type\": \"text\"}\n"
            "\n"
            "输入：\"添加我的生日，日期是1990-05-15\"\n"
            "输出：{\"intent\": \"add_custom\", \"festival_name\": \"我的生日\", \"date\": \"1990-05-15\", \"reminder_type\": \"text\"}"
        )

        user_prompt = f"请解析这句话的节日相关意图：{text}"

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            content = resp.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                return data

        except Exception as e:
            print(f"[FestivalLLM] 解析失败: {e}")

        return None


class FestivalReminderGenerator:
    """节日提醒内容生成器"""

    def __init__(self):
        self.llm_parser = FestivalLLMParser()

    def generate_reminder(self, festival: FestivalItem) -> str:
        """生成节日提醒内容"""
        if festival.reminder_type == "text" and festival.reminder_content:
            # 文字提醒
            return f"节日提醒：今天是{festival.name}！{festival.reminder_content}"

        elif festival.reminder_type == "song":
            # 唱歌提醒（这里可以调用TTS的唱歌功能，或生成歌词）
            song_content = festival.reminder_content or self._get_default_song(festival.name)
            return f"节日快乐歌：{song_content}"

        elif festival.reminder_type == "poem":
            # 诗朗诵
            poem_content = festival.reminder_content or self._get_default_poem(festival.name)
            return f"节日诗朗诵：{poem_content}"

        elif festival.reminder_type == "custom":
            # 自定义提醒
            return festival.reminder_content or f"今天是{festival.name}，祝您节日快乐！"

        else:
            # 默认文字提醒
            return f"节日提醒：今天是{festival.name}！祝您节日快乐！"

    def _get_default_song(self, festival_name: str) -> str:
        """获取默认歌曲"""
        song_map = {
            "春节": "春节快乐歌：新春佳节到，阖家欢乐好，万事如意步步高！",
            "中秋节": "中秋节快乐歌：月儿圆圆照九州，佳节团圆人欢喜，中秋快乐祝福你！",
            "生日": "生日快乐歌：祝你生日快乐，祝你生日快乐，祝你生日快乐！"
        }
        return song_map.get(festival_name, f"{festival_name}快乐歌：祝您{festival_name}快乐！")

    def _get_default_poem(self, festival_name: str) -> str:
        """获取默认诗歌"""
        poem_map = {
            "春节": "爆竹声中一岁除，春风送暖入屠苏。千门万户曈曈日，总把新桃换旧符。",
            "中秋节": "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。",
            "生日": "人生易老天难老，岁岁重阳，今又重阳。战地黄花分外香。"
        }
        return poem_map.get(festival_name, f"{festival_name}诗：节日快乐，万事如意。")

    def get_today_reminders(self, today: date, reminder_history: set) -> List[str]:
        """获取今天的节日提醒（避免重复）"""
        manager = FestivalManager()
        today_festivals = manager.get_today_festivals(today)

        reminders = []
        for festival in today_festivals:
            # 检查是否已经提醒过
            reminder_key = f"{today.isoformat()}_{festival.id}"
            if reminder_key not in reminder_history:
                reminder_content = self.generate_reminder(festival)
                reminders.append(reminder_content)
                reminder_history.add(reminder_key)

        return reminders

