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

        if intent:
            city = (intent.get("city") or "").strip()
            days = int(intent.get("days") or 1)
            print(f"[Weather] LLM解析结果: city='{city}', days={days}")  # 调试输出
        else:
            # 2. LLM 不可用或解析失败时，退回到本地规则
            city = self._extract_city(text)
            days = self._extract_days(text)
            print(f"[Weather] LLM解析失败，退回本地规则: city='{city}', days={days}")  # 调试输出

        if not city:
            # 没提城市时，可以默认一个城市或提示用户
            return "请告诉我要查询哪座城市的天气，比如：查询上海未来三天天气。"

        forecasts = self.api.get_daily_forecast(city, days)
        if not forecasts:
            return f"暂时没有查到{city}的天气信息，请稍后再试。"

        # 组装播报文本
        lines = []
        for f in forecasts:
            line = (
                f"{f.date}，白天{f.text_day}，夜间{f.text_night}，"
                f"最高气温{f.high}度，最低气温{f.low}度，"
                f"相对湿度{f.humidity}%，风向{f.wind_direction}。"
            )
            lines.append(line)

        if len(forecasts) == 1:
            prefix = f"{city}今天的天气是："
        else:
            prefix = f"{city}未来{len(forecasts)}天的天气是："

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


