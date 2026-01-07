"""
新闻查询功能对外接口
"""

from typing import Optional, List, Dict, Any
from .news_core import NewsAPIManager, NewsContentProcessor, NewsLLMParser


class NewsCommandHandler:
    """
    新闻查询命令处理器
    支持自然语言查询新闻和生活tip
    """

    def __init__(self):
        self.api_manager = NewsAPIManager()
        self.content_processor = NewsContentProcessor()
        self.llm_parser = NewsLLMParser()

    def handle(self, text: str) -> Optional[str]:
        """
        处理新闻查询请求
        支持的查询方式：
        - "看新闻" / "新闻" - 查看时事新闻
        - "职场新闻" / "科技新闻" - 查看职场相关新闻
        - "生活新闻" - 查看生活类新闻
        - "生活小贴士" / "生活tip" - 获取生活建议
        - "职场建议" / "工作tip" - 获取职场建议
        """
        text = text.strip()
        if not text:
            return None

        # 解析用户意图
        intent_data = self.llm_parser.parse(text)
        if not intent_data:
            # 如果LLM解析失败，使用简单关键词匹配
            return self._handle_simple_keywords(text)

        intent = intent_data.get("intent")
        category = intent_data.get("category", "general")
        tip_type = intent_data.get("tip_type", "general")

        if intent == "query_news":
            return self._handle_news_query(category)
        elif intent == "query_tips":
            return self._handle_tips_query(tip_type)

        return None

    def _handle_simple_keywords(self, text: str) -> Optional[str]:
        """简单的关键词匹配（当LLM不可用时使用）"""

        text_lower = text.lower()

        # 新闻查询关键词
        if any(keyword in text_lower for keyword in ["新闻", "看新闻", "今日新闻"]):
            if "时事" in text or "综合" in text:
                return self._handle_news_query("general")
            elif "职场" in text or "科技" in text or "工作" in text:
                return self._handle_news_query("tech")
            elif "生活" in text:
                return self._handle_news_query("life")
            else:
                return self._handle_news_query("general")

        # 生活tip关键词
        if any(keyword in text_lower for keyword in ["小贴士", "tip", "建议", "技巧"]):
            if "职场" in text or "工作" in text:
                return self._handle_tips_query("work")
            elif "健康" in text or "身体" in text:
                return self._handle_tips_query("health")
            else:
                return self._handle_tips_query("general")

        return None

    def _handle_news_query(self, category: str) -> str:
        """处理新闻查询"""
        try:
            news_list = self.api_manager.get_news_by_category(category, count=5)

            if not news_list:
                category_names = {
                    "general": "时事",
                    "tech": "职场科技",
                    "life": "生活"
                }
                category_name = category_names.get(category, "新闻")
                return f"抱歉，暂时没有找到{category_name}方面的新闻。请稍后再试。"

            return self.content_processor.format_news_response(news_list, max_count=3)

        except Exception as e:
            print(f"[News] 查询新闻失败: {e}")
            return "抱歉，新闻查询暂时不可用。请稍后再试。"

    def _handle_tips_query(self, tip_type: str) -> str:
        """处理生活tip查询"""
        try:
            tips = self.api_manager.get_life_tips(tip_type)
            return self.content_processor.format_tips_response(tips)

        except Exception as e:
            print(f"[News] 查询生活tip失败: {e}")
            return "抱歉，生活建议查询暂时不可用。请稍后再试。"
