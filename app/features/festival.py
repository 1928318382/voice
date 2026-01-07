"""
节日提醒功能对外接口
"""

from typing import Optional, List
from .festival_core import FestivalManager, FestivalLLMParser, FestivalReminderGenerator
from datetime import date


class FestivalCommandHandler:
    """
    节日提醒命令处理器
    支持节日查询、自定义节日添加、提醒方式设置
    """

    def __init__(self):
        self.manager = FestivalManager()
        self.llm_parser = FestivalLLMParser()
        self.reminder_generator = FestivalReminderGenerator()
        self.reminder_history: set = set()  # 记录已提醒的节日

    def handle(self, text: str) -> Optional[str]:
        """
        处理节日相关指令
        支持的指令：
        - "节日" / "有哪些节日" - 查询节日
        - "添加节日" / "自定义节日" - 添加自定义节日
        - "设置提醒" / "修改提醒" - 修改节日提醒方式
        """
        text = text.strip()
        if not text:
            return None

        # 解析用户意图
        intent_data = self.llm_parser.parse_festival_command(text)
        if not intent_data:
            # 如果LLM解析失败，使用简单关键词匹配
            return self._handle_simple_keywords(text)

        intent = intent_data.get("intent")
        festival_name = intent_data.get("festival_name", "")
        date_str = intent_data.get("date", "")
        reminder_type = intent_data.get("reminder_type", "text")
        reminder_content = intent_data.get("reminder_content")

        if intent == "add_custom":
            return self._handle_add_custom_festival(festival_name, date_str, reminder_type, reminder_content)
        elif intent == "update_reminder":
            return self._handle_update_reminder(festival_name, reminder_type, reminder_content)
        elif intent == "query_festivals":
            return self._handle_query_festivals()

        return None

    def _handle_simple_keywords(self, text: str) -> Optional[str]:
        """简单的关键词匹配"""

        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["节日", "节日提醒", "有哪些节日"]):
            return self._handle_query_festivals()

        # 检查是否包含纪念日、生日等关键词
        if any(keyword in text_lower for keyword in ["纪念日", "生日", "周年"]):
            # 尝试解析并添加自定义节日
            # 提取日期和名称
            import re
            from datetime import datetime
            
            # 中文数字映射
            cn_num_map = {
                "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
                "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25, "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30, "三十一": 31
            }
            
            # 尝试提取日期（如：一月八号、1月8日、2026-01-08等）
            date_str = None
            year = datetime.now().year  # 默认使用当前年份
            
            # 先尝试提取完整日期（包含年份）
            full_date_match = re.search(r"(\d{4})[-年](\d{1,2}|[一二三四五六七八九十]+)[-月](\d{1,2}|[一二三四五六七八九十]+)[日号]?", text)
            if full_date_match:
                year = int(full_date_match.group(1))
                month_str = full_date_match.group(2)
                day_str = full_date_match.group(3)
                
                # 转换月份
                if month_str in cn_num_map:
                    month = cn_num_map[month_str]
                else:
                    month = int(month_str)
                
                # 转换日期
                if day_str in cn_num_map:
                    day = cn_num_map[day_str]
                else:
                    day = int(day_str)
                
                date_str = f"{year}-{month:02d}-{day:02d}"
            else:
                # 尝试提取只有月日的日期
                month_day_match = re.search(r"(\d{1,2}|[一二三四五六七八九十]+)月(\d{1,2}|[一二三四五六七八九十]+)[日号]", text)
                if month_day_match:
                    month_str = month_day_match.group(1)
                    day_str = month_day_match.group(2)
                    
                    # 转换月份
                    if month_str in cn_num_map:
                        month = cn_num_map[month_str]
                    else:
                        month = int(month_str)
                    
                    # 转换日期
                    if day_str in cn_num_map:
                        day = cn_num_map[day_str]
                    else:
                        day = int(day_str)
                    
                    date_str = f"{year}-{month:02d}-{day:02d}"
            
            # 提取名称（纪念日、生日等）
            name = None
            if "入团" in text and "纪念日" in text:
                name = "入团纪念日"
            elif "生日" in text:
                name = "我的生日"
            elif "纪念日" in text:
                # 尝试提取纪念日名称
                name_match = re.search(r"(.+?)纪念日", text)
                if name_match:
                    name = name_match.group(1).strip() + "纪念日"
                else:
                    name = "纪念日"
            
            if name and date_str:
                try:
                    return self._handle_add_custom_festival(name, date_str)
                except Exception as e:
                    print(f"[Festival] 添加节日失败: {e}")
            
            # 如果无法解析，提示用户
            return "请告诉我节日名称和日期，比如：添加我的生日，日期是2026-01-08"

        if any(keyword in text_lower for keyword in ["添加节日", "自定义节日", "新建节日", "设定", "设定为"]):
            # 检查是否包含日期和名称
            if "纪念日" in text or "生日" in text:
                # 尝试解析
                return self._handle_simple_keywords(text)  # 递归调用上面的逻辑
            return "请告诉我节日名称和日期，比如：添加春节，日期是2026-02-17"

        if any(keyword in text_lower for keyword in ["设置提醒", "修改提醒", "提醒方式"]):
            return "请指定要修改的节日和提醒方式，比如：设置春节的提醒为唱歌"

        return None

    def _handle_add_custom_festival(self, name: str, date_str: str, reminder_type: str = "text",
                                   reminder_content: Optional[str] = None) -> str:
        """添加自定义节日"""
        if not name or not date_str:
            return "请提供节日名称和日期，比如：添加我的生日，日期是1990-05-15"

        try:
            festival = self.manager.add_custom_festival(name, date_str, reminder_type, reminder_content)
            return f"已成功添加自定义节日：{festival.name}，日期：{festival.custom_date}"

        except ValueError as e:
            return f"添加节日失败：{e}"

    def _handle_update_reminder(self, festival_name: str, reminder_type: str,
                               reminder_content: Optional[str] = None) -> str:
        """更新节日提醒方式"""
        # 查找节日
        target_festival = None
        for festival in self.manager.list_all_festivals():
            if festival.name == festival_name or festival.id == festival_name:
                target_festival = festival
                break

        if not target_festival:
            return f"找不到节日：{festival_name}"

        try:
            success = self.manager.update_festival_reminder(
                target_festival.id, reminder_type, reminder_content
            )
            if success:
                return f"已更新{festival_name}的提醒方式为：{reminder_type}"
            else:
                return f"更新{festival_name}提醒失败"

        except Exception as e:
            return f"更新提醒失败：{e}"

    def _handle_query_festivals(self) -> str:
        """查询所有节日"""
        festivals = self.manager.list_all_festivals()

        if not festivals:
            return "目前没有节日信息"

        # 按类型分组
        traditional = [f for f in festivals if f.type == "traditional"]
        solar_terms = [f for f in festivals if f.type == "solar_term"]
        custom = [f for f in festivals if f.type == "custom"]

        response = "节日信息：\n\n"

        if traditional:
            response += "📅 传统节日：\n"
            for festival in traditional[:5]:  # 只显示前5个
                date_info = self._get_festival_date_info(festival)
                response += f"  {festival.name}：{date_info}\n"
            if len(traditional) > 5:
                response += f"  ...还有{len(traditional) - 5}个传统节日\n"

        if solar_terms:
            response += "\n🌤️ 节气：\n"
            for festival in solar_terms[:3]:
                date_info = self._get_festival_date_info(festival)
                response += f"  {festival.name}：{date_info}\n"

        if custom:
            response += "\n🎉 自定义节日：\n"
            for festival in custom:
                date_info = self._get_festival_date_info(festival)
                response += f"  {festival.name}：{date_info}\n"

        return response

    def _get_festival_date_info(self, festival) -> str:
        """获取节日的日期信息"""
        if festival.date_type == "fixed" and festival.month and festival.day:
            return f"{festival.month}月{festival.day}日"
        elif festival.date_type == "lunar" and festival.lunar_month and festival.lunar_day:
            return f"农历{festival.lunar_month}月{festival.lunar_day}日"
        elif festival.date_type == "custom_date" and festival.custom_date:
            return festival.custom_date
        else:
            return "日期未设置"

    def get_today_festival_reminders(self) -> List[str]:
        """获取今天的节日提醒（用于首次唤醒时调用）"""
        today = date.today()
        return self.reminder_generator.get_today_reminders(today, self.reminder_history)

    def check_and_remind_festivals(self) -> Optional[str]:
        """检查是否有节日需要提醒（用于首次唤醒）"""
        reminders = self.get_today_festival_reminders()
        if reminders:
            # 合并多个节日提醒
            if len(reminders) == 1:
                return reminders[0]
            else:
                combined = "今天有多个节日：\n" + "\n".join(reminders)
                return combined

        return None