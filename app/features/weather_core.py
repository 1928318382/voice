"""
天气查询核心模块（心知天气）
支持按城市和天数查询，返回结构化预报数据：
- 日期
- 天气状况
- 最高/最低温度
- 湿度
- 风向
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests

from app.core.config import SENIVERSE_API_KEY, SENIVERSE_BASE_URL, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


@dataclass
class WeatherForecast:
    date: str          # YYYY-MM-DD
    text_day: str      # 白天天气状况
    text_night: str    # 夜间天气状况
    high: str          # 最高温度(℃)
    low: str           # 最低温度(℃)
    humidity: str      # 相对湿度(%)
    wind_direction: str  # 风向


class SeniverseWeatherAPI:
    """心知天气 API 封装"""

    def __init__(self):
        self.api_key = SENIVERSE_API_KEY
        self.base_url = SENIVERSE_BASE_URL.rstrip("/")

    def get_daily_forecast(self, location: str, days: int = 3) -> List[WeatherForecast]:
        """
        获取未来 N 天预报（含今天）
        :param location: 城市名称或城市ID（如 "beijing", "shanghai", "广州"）
        :param days: 1-7 天
        """
        days = max(1, min(days, 7))

        if not self.api_key or self.api_key == "YOUR_SENIVERSE_API_KEY":
            print("[Weather] 未配置有效的 SENIVERSE_API_KEY")
            return []

        url = f"{self.base_url}/weather/daily.json"
        params = {
            "key": self.api_key,
            "location": location,
            "language": "zh-Hans",
            "unit": "c",
            "start": 0,
            "days": days,
        }

        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results") or []
            if not results:
                print(f"[Weather] API返回空结果: {data}")
                return []

            daily_list = results[0].get("daily") or []
            forecasts: List[WeatherForecast] = []

            for item in daily_list:
                # 心知返回日期通常为 YYYY-MM-DD
                date_str = item.get("date")
                if not date_str:
                    # 兜底为今天
                    date_str = datetime.now().strftime("%Y-%m-%d")

                forecast = WeatherForecast(
                    date=date_str,
                    text_day=item.get("text_day", ""),
                    text_night=item.get("text_night", ""),
                    high=item.get("high", ""),
                    low=item.get("low", ""),
                    humidity=item.get("humidity", ""),
                    wind_direction=item.get("wind_direction", ""),
                )
                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            print(f"[Weather] 调用心知天气失败: {e}")
            return []


def forecast_to_dicts(forecasts: List[WeatherForecast]) -> List[Dict[str, Any]]:
    """便于序列化 / 调试的结构化转换"""
    return [
        {
            "date": f.date,
            "text_day": f.text_day,
            "text_night": f.text_night,
            "high": f.high,
            "low": f.low,
            "humidity": f.humidity,
            "wind_direction": f.wind_direction,
        }
        for f in forecasts
    ]


class WeatherLLMParser:
    """使用 LLM 解析天气查询意图（城市 + 天数）"""

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
        返回:
        {
          "city": "上海",
          "days": 3
        }
        """
        client = self._ensure_client()
        if client is None:
            return None

        system_prompt = (
            "你是一个天气查询意图解析助手。\n"
            "用户会用中文询问天气，请准确提取两部分信息：\n"
            "1. 城市名称：只提取城市名，不要包含时间、动作词等其他信息\n"
            "2. 天数：从句子中提取要查询的天数，1-7之间的整数\n"
            "\n"
            "规则：\n"
            "- 如果没说城市，默认城市为\"北京\"\n"
            "- 如果没说天数，默认天数为1\n"
            "- 城市名示例：上海、北京、广州、深圳、杭州\n"
            "- 天数解析：三天=3、未来一周=7、一周=7、明后天=2、明天=1、几天=3\n"
            "\n"
            "示例：\n"
            "输入：上海天气 → {\"city\": \"上海\", \"days\": 1}\n"
            "输入：北京未来三天天气 → {\"city\": \"北京\", \"days\": 3}\n"
            "输入：查询广州明后天的天气 → {\"city\": \"广州\", \"days\": 2}\n"
            "输入：深圳一周天气 → {\"city\": \"深圳\", \"days\": 7}\n"
            "\n"
            "请返回严格的JSON格式，不要其他内容：\n"
            '{"city": "城市名", "days": 数字}'
        )

        user_prompt = f"请解析这句话的天气查询意图：{text}"

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=120,
            )

            content = resp.choices[0].message.content.strip()
            print(f"[WeatherLLM] LLM原始响应: {content}")  # 调试输出
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                import json

                json_str = content[start : end + 1]
                print(f"[WeatherLLM] LLM提取的JSON: {json_str}")  # 调试输出
                data = json.loads(json_str)
                # 规范化 days 范围
                days = int(data.get("days", 1))
                data["days"] = max(1, min(days, 7))
                print(f"[WeatherLLM] LLM解析结果: city='{data.get('city', '')}', days={data['days']}")  # 调试输出
                return data
        except Exception as e:
            print(f"[WeatherLLM] 解析失败: {e}")

        return None



