"""
新闻查询核心模块
支持时事、职场、生活三个方面的新闻，以及生活tip
使用天行数据API获取新闻内容
"""

import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


@dataclass
class NewsItem:
    """新闻条目"""
    title: str
    description: str
    url: str
    source: str
    category: str
    publish_time: str


class NewsAPIManager:
    """新闻API管理器"""

    def __init__(self):
        # 天行数据API配置 - 请在下方填入你的API Key
        self.tianapi_key = "2e45b20da8276955440ab465afa87339"  # 替换为你的天行数据API Key
        self.base_url = "https://apis.tianapi.com"

        # 获取API Key的步骤：
        # 1. 访问 https://tianapi.com/
        # 2. 注册账号并登录
        # 3. 在控制台获取API Key
        # 4. 将上面的 "YOUR_TIANAPI_KEY" 替换为你的真实Key

        # 备用API配置
        self.juhe_key = "YOUR_JUHE_KEY"  # 聚合数据API
        self.juhe_base_url = "https://v.juhe.cn"

    def get_news_by_category(self, category: str, count: int = 10) -> List[NewsItem]:
        """
        根据分类获取新闻
        category: "general" (时事), "tech" (职场), "life" (生活)
        """
        try:
            # 映射到天行数据API的分类
            # 天行数据各接口 path 以官网文档为准；为兼容不同命名做多路尝试
            category_map = {
                "general": ["generalnews"],                # 综合新闻
                # 科技新闻：官方有的叫 keji，也有 IT/technews，逐个尝试
                "tech": ["keji", "it", "technews", "tech"],
                "life": ["healthskill"],                  # 健康小妙招
            }

            api_candidates = category_map.get(category, ["generalnews"])

            data = None
            last_error = None
            for api_category in api_candidates:
                url = f"{self.base_url}/{api_category}/index"
                params = {
                    "key": self.tianapi_key,
                    "num": count
                }
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    last_error = e
                    continue

            if data is None:
                raise last_error or Exception("API调用失败")

            # 天行数据新版返回格式一般为：
            # {"code":200,"msg":"success","result":{"newslist":[...]}}
            if data.get("code") == 200:
                result = data.get("result") or {}
                raw_list = result.get("newslist", []) or result.get("list", [])

                news_list: List[NewsItem] = []
                for item in raw_list:
                    # 兼容不同字段名
                    title = item.get("title") or item.get("content") or ""
                    desc = (
                        item.get("description")
                        or item.get("intro")
                        or item.get("digest")
                        or item.get("content")
                        or ""
                    )
                    url_val = item.get("url") or item.get("link") or ""
                    source = item.get("source") or item.get("infoSource") or "未知"
                    publish_time = (
                        item.get("ctime")
                        or item.get("pubtime")
                        or item.get("pubTime")
                        or item.get("date")
                        or ""
                    )

                    news_item = NewsItem(
                        title=title,
                        description=desc,
                        url=url_val,
                        source=source,
                        category=category,
                        publish_time=publish_time,
                    )
                    news_list.append(news_item)

                if not news_list:
                    print(f"[News] API返回成功但未找到 newslist，原始数据: {data}")
                return news_list
            else:
                print(f"[News] API返回错误: code={data.get('code')}, msg={data.get('msg')}")

        except Exception as e:
            print(f"[News] API调用失败: {e}")

        return []

    def get_life_tips(self, tip_type: str = "general") -> List[Dict[str, str]]:
        """
        获取生活tip
        tip_type: "general", "work", "health", "daily"
        - 通用/健康类：调用天行“健康小提示”接口（healthtip）
        - 职场类：保留本地静态提示（天行暂无线上职场建议接口）
        """
        # 职场类暂用本地数据
        if tip_type == "work":
            return [
                {"title": "高效工作小技巧", "content": "使用番茄工作法，每25分钟专注工作，休息5分钟"},
                {"title": "职场沟通要点", "content": "在会议中先倾听他人观点，再表达自己的想法"},
                {"title": "时间管理建议", "content": "每天早上规划一天的工作任务，按优先级排序"},
            ]

        # 其他类型调用天行健康小提示
        try:
            url = f"{self.base_url}/healthtip/index"
            params = {
                "key": self.tianapi_key,
                "num": 5,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 200:
                result = data.get("result") or {}
                raw_list = result.get("newslist", []) or result.get("list", [])

                tips: List[Dict[str, str]] = []

                # 情况1：返回 newslist/list 数组
                for item in raw_list:
                    title = item.get("title") or item.get("name") or "健康提示"
                    content = item.get("content") or item.get("desc") or item.get("description") or ""
                    tips.append({"title": title, "content": content})

                # 情况2：只返回单条 content 字符串（无列表）
                if not tips:
                    single_content = result.get("content")
                    if isinstance(single_content, str) and single_content.strip():
                        tips.append({"title": "健康提示", "content": single_content.strip()})

                if tips:
                    return tips
                print(f"[News] healthtip 返回空列表，原始数据: {data}")
            else:
                print(f"[News] healthtip 调用失败: code={data.get('code')}, msg={data.get('msg')}")
        except Exception as e:
            print(f"[News] 调用健康小提示失败: {e}")

        # 兜底静态数据
        return [
            {"title": "日常生活小窍门", "content": "定期整理衣柜，只保留最近一年穿过的衣服"},
            {"title": "节约用水技巧", "content": "洗澡时间控制在5-10分钟，使用节水淋浴头"},
            {"title": "环保生活习惯", "content": "随手关灯，使用可重复使用的购物袋"},
        ]


class NewsLLMParser:
    """使用LLM解析新闻查询意图"""

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
        解析新闻查询意图
        返回: {"intent": "query_news"|"query_tips", "category": "general"|"tech"|"life", "tip_type": "..."}
        """
        client = self._ensure_client()
        if client is None:
            return None

        system_prompt = (
            "你是一个新闻查询助手，请解析用户的查询意图。\n"
            "用户可能想查询新闻或获取生活建议。\n"
            "返回JSON格式：\n"
            '{"intent": "query_news" | "query_tips", "category": "general"|"tech"|"life", "tip_type": "general"|"work"|"health"|"daily"}'
        )

        user_prompt = f"请解析这句话的意图：{text}"

        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            content = resp.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                return data

        except Exception as e:
            print(f"[NewsLLM] 解析失败: {e}")

        return None


class NewsContentProcessor:
    """新闻内容处理器"""

    def __init__(self):
        self.llm_parser = NewsLLMParser()

    def summarize_news(self, news_item: NewsItem) -> str:
        """为新闻生成简短摘要"""
        # 这里可以调用LLM生成摘要，暂时返回描述
        return news_item.description[:100] + "..." if len(news_item.description) > 100 else news_item.description

    def format_news_response(self, news_list: List[NewsItem], max_count: int = 3) -> str:
        """格式化新闻回复"""
        if not news_list:
            return "抱歉，暂时没有找到相关新闻。"

        response = f"为您找到{len(news_list)}条新闻，我来为您播报前{min(max_count, len(news_list))}条：\n\n"

        for i, news in enumerate(news_list[:max_count], 1):
            summary = self.summarize_news(news)
            response += f"第{i}条：{news.title}\n{summary}\n来源：{news.source}\n\n"

        if len(news_list) > max_count:
            response += f"还有{len(news_list) - max_count}条新闻，您可以说'继续播报'来听更多。"

        return response

    def format_tips_response(self, tips: List[Dict[str, str]]) -> str:
        """格式化生活tip回复"""
        if not tips:
            return "抱歉，暂时没有相关生活建议。"

        response = f"为您准备了{len(tips)}个生活小贴士：\n\n"

        for i, tip in enumerate(tips, 1):
            response += f"{i}、{tip['title']}：{tip['content']}\n\n"

        return response
