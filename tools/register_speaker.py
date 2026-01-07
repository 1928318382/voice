#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声纹注册工具
用于注册和管理用户声纹
"""

import os
import sys
import time
import numpy as np
from typing import List, Optional

# 添加项目路径
# 添加项目路径 (根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from app.core.speaker import ECAPATDNNRecognizer
from app.core.config import SPEAKER_MIN_AUDIO_LENGTH, SAMPLE_RATE


class SpeakerRegistrationTool:
    """声纹注册工具类"""

    def __init__(self):
        self.recognizer = ECAPATDNNRecognizer()

    def record_audio(self, duration: float = 3.0) -> Optional[bytes]:
        """从麦克风录制音频"""
        try:
            import pyaudio
        except ImportError:
            print("错误：未安装pyaudio，无法录音")
            print("请运行：pip install pyaudio")
            return None

        # 增大 chunk 减少溢出风险
        chunk = 4096
        format = pyaudio.paInt16
        channels = 1
        sample_rate = SAMPLE_RATE

        p = pyaudio.PyAudio()

        try:
            # 检查是否有可用的输入设备
            info = p.get_host_api_info_by_index(0)
            device_count = info.get('deviceCount')

            input_device = None
            for i in range(device_count):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    input_device = i
                    break

            if input_device is None:
                print("错误：未找到可用的麦克风设备")
                return None

            # 增大内部缓冲区
            stream = p.open(format=format,
                          channels=channels,
                          rate=sample_rate,
                          input=True,
                          input_device_index=input_device,
                          frames_per_buffer=chunk)

            print(f"🎤 开始录音 {duration} 秒...")
            print("请清晰地说出句子，保持自然语速")

            frames = []
            
            # 使用时间控制录音，更可靠
            start_time = time.time()
            last_print = start_time
            
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
                
                # 读取可用数据，忽略溢出
                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    # 忽略读取错误，继续录音
                    pass
                
                # 每0.5秒更新一次进度
                if time.time() - last_print >= 0.5:
                    remaining = max(0, duration - elapsed)
                    print(f"\r⏱️  剩余时间: {remaining:.1f}秒", end="", flush=True)
                    last_print = time.time()

            print("\n✅ 录音完成！")

            stream.stop_stream()
            stream.close()

            audio_data = b''.join(frames)
            print(f"📊 录制了 {len(audio_data)} 字节音频")
            return audio_data

        except Exception as e:
            print(f"录音失败: {e}")
            return None
        finally:
            p.terminate()

    def generate_test_audio(self, frequency: float = 440.0, duration: float = 3.0) -> bytes:
        """生成测试音频（用于演示）"""
        sample_rate = SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # 生成简单的正弦波
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)

        # 添加一些谐波使声音更自然
        audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)

        # 添加轻微噪声
        audio += 0.05 * np.random.normal(0, 1, len(audio))

        # 转换为16位PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def register_user_interactive(self):
        """交互式注册用户"""
        print("\n" + "="*50)
        print("🎤 声纹注册系统 - 交互式注册")
        print("="*50)

        # 输入用户名
        while True:
            user_id = input("\n请输入用户名（字母、数字、下划线）：").strip()
            if not user_id:
                print("❌ 用户名不能为空")
                continue

            # 检查用户名格式
            if not all(c.isalnum() or c == '_' for c in user_id):
                print("❌ 用户名只能包含字母、数字和下划线")
                continue

            break

        # 检查用户是否已存在
        existing_users = self.recognizer.get_user_list()
        if user_id in existing_users:
            count = self.recognizer.get_user_count(user_id)
            print(f"ℹ️  用户 '{user_id}' 已存在，当前有 {count} 个样本")

            choice = input("是否继续添加新样本？(y/n): ").strip().lower()
            if choice != 'y':
                print("注册取消")
                return

        # 开始注册流程
        print(f"\n🎯 开始为用户 '{user_id}' 注册声纹")
        print("需要录制多个语音样本以提高识别准确率")
        print("建议：每个样本说不同的句子，保持自然语速")

        required_samples = 3
        collected_samples = 0
        sample_sentences = [
            "今天天气真不错",
            "我喜欢听音乐",
            "谢谢你的帮助",
            "这是一个测试句子",
            "语音识别技术很有趣"
        ]

        for i in range(required_samples):
            print(f"\n" + "-"*30)
            print(f"📝 录制第 {i+1}/{required_samples} 个样本")

            # 建议句子
            if i < len(sample_sentences):
                print(f"建议句子：'{sample_sentences[i]}'")
            else:
                print("请说任意一句自然的话")

            # 询问是否使用真实录音还是测试音频
            use_real_audio = input("使用真实录音？(y=真实录音, n=测试音频): ").strip().lower()

            if use_real_audio == 'y':
                audio_data = self.record_audio(duration=3.0)
                if audio_data is None:
                    print("❌ 录音失败，跳过此样本")
                    continue
            else:
                # 使用测试音频
                frequency = 440 + (i * 50)  # 不同的频率模拟不同语音
                audio_data = self.generate_test_audio(frequency=frequency, duration=3.0)
                print("✅ 已生成测试音频")

            # 注册样本
            success = self.recognizer.enroll_user(user_id, audio_data)

            if success:
                collected_samples += 1
                print(f"✅ 第 {i+1} 个样本注册成功！")
            else:
                print(f"❌ 第 {i+1} 个样本注册失败")

            # 如果不是最后一个样本，稍作停顿
            if i < required_samples - 1:
                time.sleep(1)

        # 注册完成
        print(f"\n" + "="*50)
        if collected_samples > 0:
            final_count = self.recognizer.get_user_count(user_id)
            print(f"🎉 注册完成！")
            print(f"👤 用户名: {user_id}")
            print(f"📊 注册样本数: {final_count}")

            if final_count >= 3:
                print("✅ 注册成功！现在可以进行声纹识别了")
            else:
                print("⚠️  建议再注册一些样本以提高准确率")
        else:
            print("❌ 注册失败，没有成功注册任何样本")

        # 显示所有用户
        self.show_registered_users()

    def register_user_batch(self, user_id: str, audio_files: List[str]):
        """批量注册用户从音频文件"""
        print(f"\n📁 批量注册用户 '{user_id}'")
        print(f"从 {len(audio_files)} 个音频文件注册")

        success_count = 0

        for i, audio_file in enumerate(audio_files, 1):
            if not os.path.exists(audio_file):
                print(f"❌ 文件不存在: {audio_file}")
                continue

            try:
                # 读取音频文件
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_file, dtype='int16')

                # 如果是立体声，转为单声道
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                # 转换为bytes
                audio_bytes = audio_data.tobytes()

                # 注册
                success = self.recognizer.enroll_user(user_id, audio_bytes)

                if success:
                    success_count += 1
                    print(f"✅ 文件 {i}/{len(audio_files)}: {os.path.basename(audio_file)} 注册成功")
                else:
                    print(f"❌ 文件 {i}/{len(audio_files)}: {os.path.basename(audio_file)} 注册失败")

            except Exception as e:
                print(f"❌ 处理文件 {audio_file} 时出错: {e}")

        print(f"\n📊 批量注册完成: {success_count}/{len(audio_files)} 个文件成功")
        final_count = self.recognizer.get_user_count(user_id)
        print(f"👤 用户 '{user_id}' 共有 {final_count} 个注册样本")

    def show_registered_users(self):
        """显示所有注册用户"""
        users = self.recognizer.get_user_list()

        if not users:
            print("\n📭 当前没有注册用户")
            return

        print(f"\n👥 已注册用户 ({len(users)} 个):")
        print("-" * 30)

        for user in users:
            count = self.recognizer.get_user_count(user)
            status = "✅" if count >= 3 else "⚠️ "
            print(f"  {status} {user}: {count} 个样本")

    def test_recognition(self):
        """测试声纹识别功能"""
        print("\n🧪 声纹识别测试")
        print("-" * 30)

        users = self.recognizer.get_user_list()
        if not users:
            print("❌ 没有注册用户，无法测试识别")
            return

        print("已注册用户:", ", ".join(users))

        # 使用测试音频进行识别
        test_frequency = 440.0  # 使用第一个用户的频率
        test_audio = self.generate_test_audio(frequency=test_frequency, duration=2.0)

        print("🎤 正在识别测试音频...")
        result = self.recognizer.identify(test_audio)

        if result != "unknown":
            print(f"✅ 识别结果: {result}")
        else:
            print("❌ 识别失败，未匹配到已知用户")

    def manage_users(self):
        """用户管理"""
        while True:
            print("\n👥 用户管理")
            print("-" * 20)
            print("1. 查看所有用户")
            print("2. 删除用户")
            print("3. 清空所有用户")
            print("4. 返回主菜单")

            choice = input("\n请选择操作 (1-4): ").strip()

            if choice == "1":
                self.show_registered_users()

            elif choice == "2":
                users = self.recognizer.get_user_list()
                if not users:
                    print("❌ 没有注册用户")
                    continue

                print("已注册用户:")
                for i, user in enumerate(users, 1):
                    count = self.recognizer.get_user_count(user)
                    print(f"  {i}. {user} ({count} 个样本)")

                try:
                    user_choice = input("请输入要删除的用户编号: ").strip()
                    user_index = int(user_choice) - 1

                    if 0 <= user_index < len(users):
                        user_to_delete = users[user_index]
                        confirm = input(f"确定要删除用户 '{user_to_delete}' 吗？(y/n): ").strip().lower()

                        if confirm == 'y':
                            success = self.recognizer.remove_user(user_to_delete)
                            if success:
                                print(f"✅ 用户 '{user_to_delete}' 已删除")
                            else:
                                print(f"❌ 删除用户 '{user_to_delete}' 失败")
                        else:
                            print("已取消删除")
                    else:
                        print("❌ 无效的用户编号")

                except ValueError:
                    print("❌ 请输入有效的数字")

            elif choice == "3":
                confirm = input("⚠️  确定要清空所有用户数据吗？此操作不可恢复！(y/n): ").strip().lower()
                if confirm == 'y':
                    success = self.recognizer.clear_database()
                    if success:
                        print("✅ 已清空所有用户数据")
                    else:
                        print("❌ 清空失败")
                else:
                    print("已取消清空操作")

            elif choice == "4":
                break

            else:
                print("❌ 无效选择，请重新输入")

    def run(self):
        """主运行函数"""
        print("🎤 声纹注册工具 v1.0")
        print("基于ECAPA-TDNN的声纹识别系统")

        while True:
            print("\n" + "="*50)
            print("主菜单")
            print("="*50)
            print("1. 📝 交互式注册用户")
            print("2. 📁 批量注册（从音频文件）")
            print("3. 👥 查看注册用户")
            print("4. 🧪 测试声纹识别")
            print("5. ⚙️  用户管理")
            print("6. 🚪 退出")

            choice = input("\n请选择功能 (1-6): ").strip()

            if choice == "1":
                self.register_user_interactive()

            elif choice == "2":
                user_id = input("请输入用户名: ").strip()
                if not user_id:
                    print("❌ 用户名不能为空")
                    continue

                files_input = input("请输入音频文件路径（用逗号分隔）: ").strip()
                if not files_input:
                    print("❌ 文件路径不能为空")
                    continue

                audio_files = [f.strip() for f in files_input.split(",") if f.strip()]
                if not audio_files:
                    print("❌ 没有有效的文件路径")
                    continue

                self.register_user_batch(user_id, audio_files)

            elif choice == "3":
                self.show_registered_users()

            elif choice == "4":
                self.test_recognition()

            elif choice == "5":
                self.manage_users()

            elif choice == "6":
                print("👋 感谢使用声纹注册工具！")
                break

            else:
                print("❌ 无效选择，请重新输入")

            # 暂停一下，让用户看到结果
            input("\n按回车键继续...")


def main():
    """主函数"""
    try:
        tool = SpeakerRegistrationTool()
        tool.run()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
