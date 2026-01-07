"""
天气查询对外接口
使用心知天气 API，支持：
- 按城市查询今日或未来几天天气（1-7天）
自然语言示例：
- “上海天气”
- “北京未来三天天气”
"""

from typing import Optional, Dict, Any
import re

from .weather_core import SeniverseWeatherAPI, WeatherLLMParser


class WeatherCommandHandler:
    def __init__(self):
        self.api = SeniverseWeatherAPI()
        self.llm_parser = WeatherLLMParser()

    def handle(self, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            return None

        # 只要包含“天气”关键词，认为是天气相关
        if "天气" not in text:
            return None

        # 1. 先尝试用 LLM 解析
        intent: Optional[Dict[str, Any]] = self.llm_parser.parse(text)
        city: Optional[str] = None
        days: int = 1
        target_day: Optional[str] = None  # 用户实际想查询的日期：今天、明天、后天等

        if intent:
            city = (intent.get("city") or "").strip()
            days = int(intent.get("days") or 1)
            # 判断用户实际想查询的日期
            text_lower = text.lower()
            if "明天" in text and "后天" not in text:
                target_day = "明天"
            elif "后天" in text:
                target_day = "后天"
            elif "今天" in text or ("今天" not in text and "明天" not in text and "后天" not in text):
                target_day = "今天"
            print(f"[Weather] LLM解析结果: city='{city}', days={days}, target_day={target_day}")  # 调试输出
        else:
            # 2. LLM 不可用或解析失败时，退回到本地规则
            city = self._extract_city(text)
            days = self._extract_days(text)
            # 判断用户实际想查询的日期
            text_lower = text.lower()
            if "明天" in text and "后天" not in text:
                target_day = "明天"
            elif "后天" in text:
                target_day = "后天"
            else:
                target_day = "今天"
            print(f"[Weather] LLM解析失败，退回本地规则: city='{city}', days={days}, target_day={target_day}")  # 调试输出

        if not city:
            # 没提城市时，可以默认一个城市或提示用户
            return "请告诉我要查询哪座城市的天气，比如：查询上海未来三天天气。"

        forecasts = self.api.get_daily_forecast(city, days)
        if not forecasts:
            return f"暂时没有查到{city}的天气信息，请稍后再试。"

        # 根据用户实际想查询的日期，从forecasts中选择对应的数据
        selected_forecasts = []
        if target_day == "明天" and len(forecasts) >= 2:
            # 用户查询明天，取第二个（索引1）
            selected_forecasts = [forecasts[1]]
        elif target_day == "后天" and len(forecasts) >= 3:
            # 用户查询后天，取第三个（索引2）
            selected_forecasts = [forecasts[2]]
        elif target_day == "今天" or len(forecasts) == 1:
            # 用户查询今天，取第一个（索引0），或者只有一天数据
            selected_forecasts = [forecasts[0]]
        else:
            # 其他情况，返回所有查询到的数据
            selected_forecasts = forecasts

        # 组装播报文本
        lines = []
        for f in selected_forecasts:
            line = (
                f"{f.date}，白天{f.text_day}，夜间{f.text_night}，"
                f"最高气温{f.high}度，最低气温{f.low}度，"
                f"相对湿度{f.humidity}%，风向{f.wind_direction}。"
            )
            lines.append(line)

        # 根据选择的日期生成前缀
        if len(selected_forecasts) == 1:
            if target_day == "明天":
                prefix = f"{city}明天的天气是："
            elif target_day == "后天":
                prefix = f"{city}后天的天气是："
            else:
                prefix = f"{city}今天的天气是："
        else:
            prefix = f"{city}未来{len(selected_forecasts)}天的天气是："

        return prefix + " ".join(lines)

    def _extract_city(self, text: str) -> Optional[str]:
        """
        非严格城市解析：简单从“查询XX天气”“XX天气”“XX未来三天天气”中抽取。
        实际项目可结合城市词典或NLP。
        """
        # 常见模式：XX天气
        m = re.search(r"(.+?)天气", text)
        if m:
            city = m.group(1)
            # 去掉“查询/看看/一下”等前缀词
            city = re.sub(r"(查询|看看|看下|一下|未来|\s)", "", city)
            return city or None
        return None

    def _extract_days(self, text: str) -> int:
        """从文本中解析天数，默认 1 天，最大 7 天。"""
        # 先查数字，如“3天”“未来5天”
        m = re.search(r"(\d+)\s*天", text)
        if m:
            try:
                days = int(m.group(1))
                return max(1, min(days, 7))
            except ValueError:
                pass

        # 再查中文数字
        cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
        m2 = re.search(r"([一二三四五六七])天", text)
        if m2:
            d = cn_map.get(m2.group(1), 1)
            return d

        # 出现“未来”但没说具体天数时，给 3 天
        if "未来" in text:
            return 3

        return 1


