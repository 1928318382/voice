#!/usr/bin/env python3
"""
测试声纹注册功能
"""

import os
import sys
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.core.speaker import ECAPATDNNRecognizer

def test_basic_registration():
    """测试基本注册功能"""
    print("🧪 测试声纹注册功能")
    print("-" * 40)

    # 创建识别器
    recognizer = ECAPATDNNRecognizer()

    # 生成测试音频
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 测试音频1（用户A）
    audio1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio1_bytes = (audio1 * 32767).astype(np.int16).tobytes()

    # 测试音频2（用户B）
    audio2 = 0.5 * np.sin(2 * np.pi * 550 * t)
    audio2_bytes = (audio2 * 32767).astype(np.int16).tobytes()

    print("1. 注册用户A...")
    success1 = recognizer.enroll_user("user_a", audio1_bytes)
    print(f"   结果: {'成功' if success1 else '失败'}")

    print("2. 注册用户B...")
    success2 = recognizer.enroll_user("user_b", audio2_bytes)
    print(f"   结果: {'成功' if success2 else '失败'}")

    print("3. 查看注册用户...")
    users = recognizer.get_user_list()
    print(f"   已注册用户: {users}")

    for user in users:
        count = recognizer.get_user_count(user)
        print(f"   用户 {user}: {count} 个样本")

    print("4. 测试识别...")
    # 识别用户A的音频
    result_a = recognizer.identify(audio1_bytes)
    print(f"   音频A识别结果: {result_a}")

    # 识别用户B的音频
    result_b = recognizer.identify(audio2_bytes)
    print(f"   音频B识别结果: {result_b}")

    print("5. 测试验证...")
    # 验证用户A
    is_verified_a, confidence_a = recognizer.verify("user_a", audio1_bytes)
    print(f"   用户A验证结果: {is_verified_a} (置信度: {confidence_a:.3f})")

    # 验证用户B
    is_verified_b, confidence_b = recognizer.verify("user_b", audio2_bytes)
    print(f"   用户B验证结果: {is_verified_b} (置信度: {confidence_b:.3f})")

    print("\n✅ 声纹注册功能测试完成！")

    # 清理测试数据
    print("\n🧹 清理测试数据...")
    recognizer.clear_database()
    print("✅ 测试数据已清理")

if __name__ == "__main__":
    test_basic_registration()
