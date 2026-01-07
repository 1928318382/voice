#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能声纹注册系统 v4.0
支持音频质量监控、实时反馈、多种录音模式
"""

import os
import sys
import time
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple, Dict
import threading

# 添加项目路径 (根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from app.core.speaker import ECAPATDNNRecognizer
from app.core.config import SAMPLE_RATE, MIN_ENROLLMENT_SAMPLES, MIC_DEVICE_INDEX, SPEAKER_SIMILARITY_THRESHOLD
from app.core.enhancement import AudioEnhancer


class AudioQualityAnalyzer:
    """音频质量分析器"""

    def __init__(self):
        self.quality_thresholds = {
            'min_rms': 0.08,      # 最小有效RMS
            'max_rms': 0.7,       # 最大RMS（避免过载）
            'min_length': 1.5,    # 最短音频长度
            'max_silence': 0.2,   # 最大静音比例
        }

    def analyze_audio(self, audio_data: bytes) -> Dict:
        """分析音频质量"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            length = len(audio_np) / SAMPLE_RATE

            # 基本统计
            rms = np.sqrt(np.mean(audio_np**2))
            peak = np.max(np.abs(audio_np))
            silence_frames = np.sum(np.abs(audio_np) < 0.01)
            silence_ratio = silence_frames / len(audio_np)

            # 质量评分
            quality_score = self._calculate_quality_score(rms, length, silence_ratio)

            return {
                'rms': rms,
                'peak': peak,
                'length': length,
                'silence_ratio': silence_ratio,
                'quality_score': quality_score,
                'is_acceptable': quality_score >= 0.6,
                'feedback': self._get_feedback(rms, length, silence_ratio, quality_score)
            }
        except Exception as e:
            return {
                'error': str(e),
                'is_acceptable': False,
                'feedback': f"音频分析失败: {e}"
            }

    def _calculate_quality_score(self, rms: float, length: float, silence_ratio: float) -> float:
        """计算质量分数 (0-1)"""
        score = 0.0

        # RMS评分 (40%)
        if rms < self.quality_thresholds['min_rms']:
            rms_score = rms / self.quality_thresholds['min_rms'] * 0.5
        elif rms > self.quality_thresholds['max_rms']:
            rms_score = 0.5
        else:
            # 最佳范围 0.15-0.4
            if 0.15 <= rms <= 0.4:
                rms_score = 1.0
            else:
                rms_score = 0.8

        # 长度评分 (30%)
        if length < self.quality_thresholds['min_length']:
            length_score = length / self.quality_thresholds['min_length']
        else:
            length_score = min(length / 3.0, 1.0)

        # 静音评分 (30%)
        silence_score = 1.0 - min(silence_ratio / self.quality_thresholds['max_silence'], 1.0)

        score = 0.4 * rms_score + 0.3 * length_score + 0.3 * silence_score
        return min(max(score, 0.0), 1.0)

    def _get_feedback(self, rms: float, length: float, silence_ratio: float, score: float) -> str:
        """生成质量反馈"""
        feedback = []

        if score >= 0.8:
            feedback.append("✅ 音频质量优秀！")
        elif score >= 0.6:
            feedback.append("👍 音频质量良好")
        else:
            feedback.append("⚠️ 音频质量需要改进")

        # 具体建议
        if rms < self.quality_thresholds['min_rms']:
            feedback.append(f"声音太小 (RMS: {rms:.3f})，请靠近麦克风")
        elif rms > self.quality_thresholds['max_rms']:
            feedback.append(f"声音太大 (RMS: {rms:.3f})，请远离麦克风")
        if length < self.quality_thresholds['min_length']:
            feedback.append(f"录音太短 ({length:.1f}秒)，请说完整的句子")
        if silence_ratio > self.quality_thresholds['max_silence']:
            feedback.append(f"静音太多 ({silence_ratio:.1%})，请清晰说话")

        return " | ".join(feedback)


class SmartSpeakerRegistrationTool:
    """智能声纹注册工具 v4.0"""

    def __init__(self):
        self.recognizer = ECAPATDNNRecognizer()
        self.quality_analyzer = AudioQualityAnalyzer()
        self.enhancer = AudioEnhancer()

        # 注册会话状态
        self.session_stats = {
            'user_id': None,
            'attempts': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'best_quality': 0.0
        }

        print("🎤 智能声纹注册工具 v4.0 初始化完成")
        print("✨ 支持实时质量监控和智能引导")

    def run(self):
        """运行注册工具"""
        self.show_welcome()

        while True:
            choice = self.show_menu()

            if choice == '1':
                self.interactive_registration()
            elif choice == '2':
                self.batch_registration()
            elif choice == '3':
                self.show_registered_users()
            elif choice == '4':
                self.test_recognition()
            elif choice == '5':
                self.user_management()
            elif choice == '6':
                self.quit()
                break
            else:
                print("❌ 无效选择，请重新输入")

    def show_welcome(self):
        """显示欢迎信息"""
        print("\n" + "="*60)
        print("🎤 智能声纹注册工具 v4.0")
        print("基于ECAPA-TDNN的声纹识别系统")
        print("="*60)
        print("✨ 智能质量监控")
        print("🎯 实时反馈引导")
        print("🔄 自动重试机制")
        print("📊 详细统计报告")
        print("="*60)

    def show_menu(self):
        """显示主菜单"""
        print("\n主菜单")
        print("-"*40)
        print("1. 📝 交互式注册用户")
        print("2. 📁 批量注册（从音频文件）")
        print("3. 👥 查看注册用户")
        print("4. 🧪 测试声纹识别")
        print("5. ⚙️  用户管理")
        print("6. 🚪 退出")
        print("-"*40)

        return input("请选择功能 (1-6): ").strip()

    def interactive_registration(self):
        """智能交互式注册"""
        print("\n" + "="*60)
        print("🎤 智能声纹注册 - 交互式模式")
        print("="*60)

        # 输入用户名
        while True:
            try:
                user_id = input("\n请输入用户名（字母、数字、下划线）：").strip()
                if not user_id:
                    print("❌ 用户名不能为空")
                    continue

                # 检查用户名格式
                if not all(c.isalnum() or c == '_' for c in user_id):
                    print("❌ 用户名只能包含字母、数字和下划线")
                    continue

                break
            except KeyboardInterrupt:
                print("\n❌ 注册取消")
                return

        self.session_stats['user_id'] = user_id

        # 检查用户是否已存在
        existing_users = self.recognizer.get_user_list()
        if user_id in existing_users:
            count = self.recognizer.get_user_count(user_id)
            print(f"ℹ️  用户 '{user_id}' 已存在，当前有 {count} 个样本")

            choice = input("是否继续添加新样本？(y/n): ").strip().lower()
            if choice != 'y':
                print("注册取消")
                return

        # 开始智能注册流程
        self.smart_registration_process(user_id)

    def smart_registration_process(self, user_id: str):
        """智能注册流程"""
        print(f"\n🎯 开始为用户 '{user_id}' 进行智能注册")
        print(f"需要录制 {MIN_ENROLLMENT_SAMPLES} 个高质量语音样本")
        print("\n智能提示：")
        print("• 系统会自动分析音频质量")
        print("• 质量不佳时会建议重新录制")
        print("• 每个样本录制3秒，请说完整的句子")

        # 示例句子
        sample_sentences = [
            "今天天气真不错",
            "我喜欢听音乐",
            "谢谢你的帮助",
            "这是一个测试句子",
            "语音识别技术很有趣",
            "人工智能发展很快",
            "请问现在几点了",
            "我想听一首歌",
            "这个功能很实用",
            "声纹识别真神奇"
        ]

        collected_samples = 0
        max_attempts_per_sample = 3

        while collected_samples < MIN_ENROLLMENT_SAMPLES:
            sample_attempt = 0
            sample_accepted = False

            while sample_attempt < max_attempts_per_sample and not sample_accepted:
                sample_attempt += 1

                print(f"\n📝 录制第 {collected_samples + 1}/{MIN_ENROLLMENT_SAMPLES} 个样本")
                if sample_attempt > 1:
                    print(f"🔄 第 {sample_attempt} 次尝试")

                # 显示建议句子
                if collected_samples < len(sample_sentences):
                    print(f"💬 建议句子：'{sample_sentences[collected_samples]}'")
                else:
                    print("💬 请说任意一句自然的话")

                print("\n请选择录音模式:")
                print("1. 🎤 真实录音（推荐）")
                print("2. 🔊 测试音频（用于调试）")
                print("3. ⏭️ 跳过此样本")

                choice = input("请选择 (1-3): ").strip()

                if choice == '3':
                    print("⏭️ 跳过此样本")
                    sample_accepted = True
                    collected_samples += 1
                    break
                elif choice == '2':
                    # 生成测试音频
                    frequency = 440 + (collected_samples * 50)
                    audio_data = self.generate_test_audio(frequency=frequency, duration=3.0)
                    quality_info = {'is_acceptable': True, 'quality_score': 0.8, 'feedback': '测试音频（质量固定）'}
                    print("✅ 已生成测试音频")
                elif choice == '1':
                    # 真实录音
                    audio_data, quality_info = self.record_audio_with_quality_check(duration=3.0)

                    if audio_data is None:
                        print(f"❌ 录音失败: {quality_info.get('error', '未知错误')}")
                        if sample_attempt >= max_attempts_per_sample:
                            print("⚠️ 多次录音失败，建议检查麦克风设置")
                        continue

                    # 显示质量反馈
                    print(f"📊 质量分析: {quality_info['feedback']}")

                    if not quality_info.get('is_acceptable', False):
                        print("⚠️ 音频质量不佳，建议重新录制")
                        if input("仍要使用此录音？(y/n): ").strip().lower() != 'y':
                            continue
                else:
                    print("❌ 无效选择，请重新选择")
                    continue

                # 注册样本
                success = self.recognizer.enroll_user(user_id, audio_data)
                if success:
                    collected_samples += 1
                    self.session_stats['quality_scores'].append(quality_info.get('quality_score', 0))
                    print(f"✅ 第 {collected_samples}/{MIN_ENROLLMENT_SAMPLES} 个样本注册成功！")
                    sample_accepted = True
                else:
                    print("❌ 注册失败，声纹识别器可能有问题")

            if sample_attempt >= max_attempts_per_sample and not sample_accepted:
                print(f"⚠️ 第 {collected_samples + 1} 个样本多次尝试失败，跳过")
                collected_samples += 1

            # 样本间暂停
            if collected_samples < MIN_ENROLLMENT_SAMPLES:
                input("\n按回车键继续录制下一个样本...")

        # 注册完成，显示统计
        self.show_registration_summary(user_id)

    def record_audio_with_quality_check(self, duration: float = 3.0, show_feedback: bool = True) -> Tuple[Optional[bytes], Dict]:
        """智能录音函数，包含质量分析"""
        try:
            import pyaudio

            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = SAMPLE_RATE

            p = pyaudio.PyAudio()

            # 验证设备
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, MIC_DEVICE_INDEX)
                if device_info.get('maxInputChannels') <= 0:
                    return None, {'error': f'设备 {MIC_DEVICE_INDEX} 没有输入通道'}
                if show_feedback:
                    print(f"🎤 使用设备: {device_info['name']} (索引: {MIC_DEVICE_INDEX})")
            except Exception as e:
                return None, {'error': f'无法访问设备 {MIC_DEVICE_INDEX}: {e}'}

            # 打开音频流
            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=chunk
            )

            print(f"🎙️ 开始录音 {duration} 秒...")
            print("请清晰地说出句子，保持自然语速")
            print("💡 提示：保持15-30cm距离，音量适中")

            frames = []

            # 显示进度条
            total_chunks = int(duration * SAMPLE_RATE / chunk)
            for i in range(total_chunks):
                data = stream.read(chunk)
                frames.append(data)

                # 每0.5秒显示一次进度
                if i % int(0.5 * SAMPLE_RATE / chunk) == 0:
                    progress = (i + 1) / total_chunks
                    bar_length = 20
                    filled = int(bar_length * progress)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"\r⏱️ 录音进度: [{bar}] {progress:.1%}", end="", flush=True)

            print(" ✅")

            stream.stop_stream()
            stream.close()
            p.terminate()

            # 合并音频数据
            audio_data = b''.join(frames)

            # 分析音频质量
            quality_info = self.quality_analyzer.analyze_audio(audio_data)

            return audio_data, quality_info

        except Exception as e:
            print(f"❌ 录音过程中出错: {e}")
            return None, {'error': str(e)}

    def show_registration_summary(self, user_id: str):
        """显示注册总结"""
        print(f"\n🎉 智能注册完成！")
        print("="*50)

        final_count = self.recognizer.get_user_count(user_id)
        print(f"👤 用户: {user_id}")
        print(f"📊 注册样本数: {final_count}")

        if self.session_stats['quality_scores']:
            avg_quality = np.mean(self.session_stats['quality_scores'])
            best_quality = np.max(self.session_stats['quality_scores'])

            print(f"📈 平均质量分数: {avg_quality:.2f}/1.0")
            print(f"🏆 最佳质量分数: {best_quality:.2f}/1.0")

            if avg_quality >= 0.8:
                print("🎯 注册质量: 优秀")
            elif avg_quality >= 0.6:
                print("👍 注册质量: 良好")
            else:
                print("⚠️ 注册质量: 需要改进")

        print("\n💡 使用建议:")
        print("• 现在可以测试声纹识别了")
        print("• 建议在不同环境下多测试几次")
        print("• 定期更新注册样本以提高准确性")

        # 重置会话统计
        self.session_stats = {
            'user_id': None,
            'attempts': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'best_quality': 0.0
        }

    def generate_test_audio(self, frequency: float = 440.0, duration: float = 3.0) -> bytes:
        """生成测试音频"""
        sample_rate = SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # 生成带有谐波的音频信号
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        audio += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # 二倍频
        audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # 三倍频

        # 添加少量噪声
        audio += 0.05 * np.random.normal(0, 1, len(audio))

        # 转换为16位整数
        audio_int16 = (audio * 32767).astype(np.int16)

        return audio_int16.tobytes()

    def batch_registration(self):
        """批量注册功能"""
        print("\n📁 批量注册功能")
        print("此功能用于从音频文件批量注册用户")
        print("暂未实现，敬请期待...")

        input("\n按回车键返回...")

    def show_registered_users(self):
        """显示注册用户"""
        users = self.recognizer.get_user_list()

        print(f"\n👥 已注册用户 (共 {len(users)} 个)")
        print("="*50)

        if not users:
            print("📭 暂无注册用户")
            return

        total_samples = 0
        for i, user in enumerate(users, 1):
            sample_count = self.recognizer.get_user_count(user)
            total_samples += sample_count

            # 根据样本数显示状态
            if sample_count >= MIN_ENROLLMENT_SAMPLES:
                status = "✅"
            elif sample_count > 0:
                status = "⚠️ "
            else:
                status = "❌"

            print(f"  {i}. {user} ({sample_count}个样本)")  # Fixed 'count' to 'sample_count'
        print("="*50)
        print(f"📊 总样本数: {total_samples}")
        print(f"📈 平均样本数: {total_samples/max(len(users), 1):.1f}")
        input("\n按回车键返回...")

    def test_recognition(self):
        """测试声纹识别"""
        users = self.recognizer.get_user_list()

        if not users:
            print("\n❌ 没有注册用户，无法测试")
            input("按回车键返回...")
            return

        print(f"\n🧪 声纹识别测试 (已注册用户: {len(users)}个)")
        print("请选择测试模式:")
        print("1. 🎤 实时录音测试")
        print("2. 📁 文件测试（暂未实现）")

        choice = input("请选择 (1-2): ").strip()

        if choice == '1':
            self.real_time_recognition_test()
        else:
            print("暂未实现")
            input("按回车键返回...")

    def real_time_recognition_test(self):
        """实时识别测试"""
        print("\n🎤 实时声纹识别测试")
        print("按说明进行操作...")

        # 这里可以实现实时测试逻辑
        # 暂时使用简化的测试

        audio_data, quality_info = self.record_audio_with_quality_check(duration=3.0)

        if audio_data is None:
            print("❌ 录音失败")
            return

        # 进行识别
        user_id = self.recognizer.identify(audio_data)

        if user_id != "unknown":
            print(f"✅ 识别结果: {user_id}")
        else:
            print("❌ 未识别到已知用户")

        input("\n按回车键返回...")

    def user_management(self):
        """用户管理"""
        while True:
            print("\n⚙️ 用户管理")
            print("-"*30)
            users = self.recognizer.get_user_list()

            if users:
                print("已注册用户:")
                for i, user in enumerate(users, 1):
                    count = self.recognizer.get_user_count(user)
                    print(f"  {i}. {user} ({count}个样本)")
            else:
                print("暂无注册用户")

            print("\n操作选项:")
            print("1. 🗑️ 删除用户")
            print("2. 📊 查看用户详情")
            print("3. 🔄 重新注册用户")
            print("4. ↩️ 返回主菜单")

            choice = input("请选择 (1-4): ").strip()

            if choice == '1' and users:
                self.delete_user(users)
            elif choice == '2' and users:
                self.show_user_details(users)
            elif choice == '3':
                self.interactive_registration()
            elif choice == '4':
                break
            else:
                print("❌ 无效选择")

    def delete_user(self, users):
        """删除用户"""
        try:
            idx = int(input("输入要删除的用户编号: ")) - 1
            if 0 <= idx < len(users):
                user_id = users[idx]
                confirm = input(f"确认删除用户 '{user_id}' 吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    if self.recognizer.remove_user(user_id):
                        print(f"✅ 用户 '{user_id}' 已删除")
                    else:
                        print("❌ 删除失败")
            else:
                print("❌ 无效用户编号")
        except ValueError:
            print("❌ 请输入有效的数字")

    def show_user_details(self, users):
        """显示用户详情"""
        try:
            idx = int(input("输入要查看的用户编号: ")) - 1
            if 0 <= idx < len(users):
                user_id = users[idx]
                count = self.recognizer.get_user_count(user_id)
                print(f"\n👤 用户详情: {user_id}")
                print(f"📊 样本数量: {count}")
                print(f"📈 注册状态: {'✅ 完整' if count >= MIN_ENROLLMENT_SAMPLES else '⚠️ 不完整'}")
            else:
                print("❌ 无效用户编号")
        except ValueError:
            print("❌ 请输入有效的数字")

        input("\n按回车键返回...")

    def quit(self):
        """退出程序"""
        print("\n👋 感谢使用智能声纹注册工具！")
        print("🎤 再见！")


def main():
    """主函数"""
    try:
        tool = SmartSpeakerRegistrationTool()
        tool.run()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
