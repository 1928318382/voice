import numpy as np
import librosa
from scipy import signal
from typing import Optional, Tuple
import os
import tempfile
import subprocess
import sys

# 可选导入 (如果安装了相应库)
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

try:
    import rnnoise
    HAS_RNNOISE = True
except ImportError:
    HAS_RNNOISE = False

from .config import (
    ENHANCEMENT_MODE, NOISE_REDUCTION_STRENGTH, MIN_VOICE_RATIO,
    AGC_TARGET_LEVEL, AGC_MAX_GAIN, VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION, SAMPLE_RATE
)


class AudioEnhancer:
    """
    语音增强模块 - 多级增强流水线
    支持：降噪、AGC、VAD、噪声门等
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.mode = ENHANCEMENT_MODE

        # 初始化各个处理模块
        self.vad = None
        self.rnnoise_state = None
        self._vad_initialized = False  # 标记VAD是否已初始化

        if self.mode in ["basic", "advanced", "full"]:
            print(f"[Enhancement] 初始化语音增强模块 (模式: {self.mode})")

            # VAD将在第一次使用时延迟初始化（避免Windows多进程pickle问题）
            if HAS_WEBRTCVAD and self.mode in ["advanced", "full"]:
                print("[Enhancement] VAD模块将在首次使用时初始化")

            # 初始化RNNoise (暂时跳过，因为rnnoise-python可能有兼容性问题)
            # if HAS_RNNOISE and self.mode in ["advanced", "full"]:
            #     try:
            #         self.rnnoise_state = rnnoise.create()
            #         print("[Enhancement] RNNoise降噪已启用")
            #     except Exception as e:
            #         print(f"[Enhancement] RNNoise初始化失败: {e}")
            #         self.rnnoise_state = None
        else:
            print("[Enhancement] 语音增强已禁用")

    def process(self, audio_data: bytes) -> bytes:
        """
        音频增强处理流水线
        """
        if self.mode == "none":
            return audio_data

        try:
            # 转换为numpy数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # 归一化到[-1, 1]

            # 1. 预处理：直流偏移校正
            audio_np = self._remove_dc_offset(audio_np)

            # 2. 语音活动检测 (VAD)
            if self.vad and self.mode in ["advanced", "full"]:
                is_speech = self._is_speech(audio_np)
                if not is_speech:
                    # 非语音段，应用更强的降噪
                    audio_np = self._apply_noise_gate(audio_np, threshold=0.01)

            # 3. 降噪处理
            if HAS_NOISEREDUCE and self.mode in ["basic", "advanced", "full"]:
                audio_np = self._reduce_noise(audio_np)

            # 4. RNNoise深度降噪 (暂时跳过)
            # if self.rnnoise_state and self.mode in ["advanced", "full"]:
            #     audio_np = self._apply_rnnoise(audio_np)

            # 5. 自动增益控制 (AGC)
            if self.mode in ["basic", "advanced", "full"]:
                audio_np = self._apply_agc(audio_np)

            # 6. 后处理：限幅和归一化
            audio_np = self._post_process(audio_np)

            # 转换回bytes
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()

        except Exception as e:
            print(f"[Enhancement] 处理出错，回退到原始音频: {e}")
            return audio_data

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """移除直流偏移"""
        return audio - np.mean(audio)

    def _is_speech(self, audio: np.ndarray) -> bool:
        """语音活动检测"""
        # 延迟初始化VAD（避免Windows多进程pickle问题）
        if not self._vad_initialized and HAS_WEBRTCVAD and self.mode in ["advanced", "full"]:
            try:
                self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
                self._vad_initialized = True
                print("[Enhancement] VAD模块延迟初始化成功")
            except Exception as e:
                print(f"[Enhancement] VAD延迟初始化失败: {e}")
                self.vad = None
                return True

        if not self.vad:
            return True

        try:
            # 转换为16位PCM
            audio_int16 = (audio * 32767).astype(np.int16).tobytes()

            # VAD需要10ms, 20ms, 或30ms的帧
            frame_size = int(self.sample_rate * VAD_FRAME_DURATION / 1000)
            if len(audio_int16) < frame_size * 2:
                return True

            # 检查多个帧
            speech_frames = 0
            total_frames = 0

            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                    total_frames += 1

            # 如果超过50%的帧被识别为语音，则认为是语音
            return speech_frames / max(total_frames, 1) > 0.5

        except Exception as e:
            print(f"[Enhancement] VAD检测失败: {e}")
            return True

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """使用noisereduce进行降噪"""
        if not HAS_NOISEREDUCE:
            return audio

        try:
            # 使用前20%的音频作为噪声样本
            noise_sample_size = min(len(audio) // 5, int(self.sample_rate * 0.5))
            if noise_sample_size > 100:
                noise_sample = audio[:noise_sample_size]
                return nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=NOISE_REDUCTION_STRENGTH
                )
            else:
                # 音频太短，直接降噪
                return nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    prop_decrease=NOISE_REDUCTION_STRENGTH * 0.5
                )
        except Exception as e:
            print(f"[Enhancement] 降噪处理失败: {e}")
            return audio

    def _apply_rnnoise(self, audio: np.ndarray) -> np.ndarray:
        """使用RNNoise进行深度降噪 (暂时跳过实现)"""
        # 由于rnnoise-python库可能有兼容性问题，先跳过
        # 未来可以考虑使用其他替代方案
        return audio

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """自动增益控制"""
        try:
            # 计算当前RMS电平
            rms = np.sqrt(np.mean(audio ** 2))

            # 目标RMS电平
            target_rms = AGC_TARGET_LEVEL

            # 计算增益
            if rms > 0:
                gain = min(target_rms / rms, AGC_MAX_GAIN)  # 最大增益限制
                audio = audio * gain

            return audio

        except Exception as e:
            return audio

    def _apply_noise_gate(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """噪声门：低于阈值的信号设为0"""
        try:
            # 计算短时能量
            frame_length = int(self.sample_rate * 0.025)  # 25ms帧
            hop_length = frame_length // 2

            energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # 扩展能量到原始长度
            energy_expanded = np.repeat(energy, hop_length)[:len(audio)]

            # 应用噪声门
            gate = (energy_expanded > threshold).astype(np.float32)

            # 平滑过渡
            gate = signal.convolve(gate, np.ones(100)/100, mode='same')
            gate = np.clip(gate, 0, 1)

            return audio * gate

        except Exception as e:
            return audio

    def _post_process(self, audio: np.ndarray) -> np.ndarray:
        """后处理：限幅和压缩"""
        try:
            # 硬限幅
            audio = np.clip(audio, -0.95, 0.95)

            # 轻微压缩 (可选)
            # 这里可以添加更复杂的压缩算法

            return audio

        except Exception as e:
            return audio


# === 测试函数 ===
def test_enhancement():
    """测试语音增强功能"""
    print("=== 语音增强模块测试 ===")

    enhancer = AudioEnhancer()

    # 生成测试音频：语音 + 噪声
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 模拟语音信号
    speech = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz音调
    speech += 0.2 * np.sin(2 * np.pi * 880 * t)  # 880Hz谐波

    # 添加噪声
    noise = 0.1 * np.random.normal(0, 1, len(t))  # 白噪声
    test_audio = speech + noise

    print(f"原始音频RMS: {np.sqrt(np.mean(test_audio ** 2)):.4f}")

    # 处理音频
    test_bytes = (test_audio * 32767).astype(np.int16).tobytes()
    enhanced_bytes = enhancer.process(test_bytes)

    # 转换回numpy
    enhanced_audio = np.frombuffer(enhanced_bytes, dtype=np.int16).astype(np.int16).astype(np.float32) / 32768.0
    print(f"增强后音频RMS: {np.sqrt(np.mean(enhanced_audio ** 2)):.4f}")

    print("语音增强测试完成")
    return True


def test_integration():
    """测试与ASR模块的集成"""
    print("=== 语音增强集成测试 ===")

    try:
        print(f"配置模式: {ENHANCEMENT_MODE}")

        # 创建增强器
        enhancer = AudioEnhancer()
        print("✓ 增强器创建成功")

        # 测试音频处理
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # 生成测试音频
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 1, len(t))
        test_bytes = (test_audio * 32767).astype(np.int16).tobytes()

        # 处理音频
        enhanced_bytes = enhancer.process(test_bytes)
        print("✓ 音频处理成功")

        # 验证输出格式
        if len(enhanced_bytes) == len(test_bytes):
            print("✓ 输出格式正确")
        else:
            print(f"⚠️ 输出长度不匹配: {len(enhanced_bytes)} vs {len(test_bytes)}")

        print("语音增强集成测试完成")
        return True

    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False


def demo_enhancement():
    """演示语音增强效果"""
    print("=== 语音增强效果演示 ===")

    enhancer = AudioEnhancer()

    # 生成测试音频：清晰语音 + 大量噪声
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 模拟清晰语音
    speech = 0.2 * np.sin(2 * np.pi * 440 * t)  # 440Hz音调
    speech += 0.15 * np.sin(2 * np.pi * 880 * t)  # 880Hz谐波

    # 添加大量噪声（模拟嘈杂环境）
    noise = 0.3 * np.random.normal(0, 1, len(t))  # 大量白噪声
    # 添加一些低频噪声（模拟风扇/空调）
    noise += 0.2 * np.sin(2 * np.pi * 50 * t)  # 50Hz低频噪声

    noisy_audio = speech + noise

    print("原始音频统计:")
    print(".4f")
    print(".4f")

    # 处理音频
    noisy_bytes = (noisy_audio * 32767).astype(np.int16).tobytes()
    enhanced_bytes = enhancer.process(noisy_bytes)

    # 转换回numpy
    enhanced_audio = np.frombuffer(enhanced_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print("\n增强后音频统计:")
    print(".4f")
    print(".4f")

    # 计算改善效果
    snr_improvement = 20 * np.log10(np.sqrt(np.mean(enhanced_audio ** 2)) / np.sqrt(np.mean((enhanced_audio - speech) ** 2))) - \
                     20 * np.log10(np.sqrt(np.mean(noisy_audio ** 2)) / np.sqrt(np.mean((noisy_audio - speech) ** 2)))

    print(f"SNR改善: {snr_improvement:.1f}dB")
    print("语音增强演示完成")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "integration":
            test_integration()
        elif sys.argv[1] == "demo":
            demo_enhancement()
        else:
            test_enhancement()
    else:
        test_enhancement()
