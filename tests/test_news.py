#!/usr/bin/env python3
"""
测试新闻查询功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.news import NewsCommandHandler

def test_news():
    """测试新闻查询功能"""
    handler = NewsCommandHandler()

    test_cases = [
        "看新闻",
        "时事新闻",
        "科技新闻",
        "生活新闻",
        "生活小贴士",
        "职场建议",
        "工作tip"
    ]

    print("测试新闻查询功能：")
    print("=" * 50)

    for test_text in test_cases:
        print(f"\n输入: {test_text}")
        try:
            result = handler.handle(test_text)
            if result:
                print(f"输出: {result[:200]}..." if len(result) > 200 else f"输出: {result}")
            else:
                print("输出: (未识别为新闻查询)")
        except Exception as e:
            print(f"错误: {e}")

        print("-" * 30)

if __name__ == "__main__":
    test_news()
