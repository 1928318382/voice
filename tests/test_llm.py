#!/usr/bin/env python3
"""
测试阿里云百炼 LLM API 是否能正常调用
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME

def test_llm_api():
    """测试 LLM API 连接和调用"""
    print("正在测试 LLM API...")
    print(f"API Key: {LLM_API_KEY[:8]}...{LLM_API_KEY[-4:] if len(LLM_API_KEY) > 12 else LLM_API_KEY}")
    print(f"Base URL: {LLM_BASE_URL}")
    print(f"Model: {LLM_MODEL_NAME}")
    print("-" * 50)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

        print("正在发送测试请求...")

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个智能助手，请简洁友好地回答用户问题。"},
                {"role": "user", "content": "你好，请介绍一下你自己。"}
            ],
            temperature=0.7,
            max_tokens=200
        )

        content = response.choices[0].message.content
        print("✅ API 调用成功！")
        print(f"模型回复: {content}")
        return True

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查 API Key 是否正确")
        print("2. 检查网络连接是否正常")
        print("3. 检查模型名称是否正确")
        print("4. 检查账户是否有余额")
        print("5. 检查 base_url 是否正确")
        return False

if __name__ == "__main__":
    success = test_llm_api()
    if success:
        print("\n🎉 大模型API配置正确，可以正常使用！")
    else:
        print("\n❌ 请检查配置后重试")
        sys.exit(1)