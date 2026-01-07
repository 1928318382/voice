#!/usr/bin/env python3
"""
测试节日提醒功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.festival import FestivalCommandHandler

def test_festival():
    """测试节日功能"""
    handler = FestivalCommandHandler()

    test_cases = [
        "有哪些节日",
        "添加我的生日，日期是1990-05-15",
        "设置春节的提醒为唱歌",
        "节日信息"
    ]

    print("测试节日提醒功能：")
    print("=" * 50)

    for test_text in test_cases:
        print(f"\n输入: {test_text}")
        try:
            result = handler.handle(test_text)
            if result:
                print(f"输出: {result[:200]}..." if len(result) > 200 else f"输出: {result}")
            else:
                print("输出: (未识别为节日命令)")
        except Exception as e:
            print(f"错误: {e}")

        print("-" * 30)

    # 测试今天的节日提醒
    print("\n测试今天的节日提醒:")
    try:
        reminders = handler.get_today_festival_reminders()
        if reminders:
            for reminder in reminders:
                print(f"提醒: {reminder}")
        else:
            print("今天没有节日")
    except Exception as e:
        print(f"获取节日提醒失败: {e}")

if __name__ == "__main__":
    test_festival()

