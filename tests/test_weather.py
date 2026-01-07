#!/usr/bin/env python3
"""
测试天气查询功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.weather import WeatherCommandHandler

def test_weather():
    """测试天气查询功能"""
    handler = WeatherCommandHandler()

    test_cases = [
        "查询上海天气",
        "北京未来三天天气",
        "看看广州这几天天气怎么样",
        "杭州明后天天气",
        "查一下南京的天气",
        "深圳未来一周天气",  # 测试边界（最多7天）
        "武汉天气",  # 简单查询
    ]

    print("测试天气查询功能：")
    print("=" * 50)

    for test_text in test_cases:
        print(f"\n输入: {test_text}")
        try:
            result = handler.handle(test_text)
            if result:
                print(f"输出: {result[:200]}..." if len(result) > 200 else f"输出: {result}")
            else:
                print("输出: (未识别为天气查询)")
        except Exception as e:
            print(f"错误: {e}")

        print("-" * 30)

if __name__ == "__main__":
    test_weather()
